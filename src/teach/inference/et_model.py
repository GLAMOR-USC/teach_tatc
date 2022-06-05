# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from modeling import constants
from modeling.datasets.tatc import TATCDataset
from modeling.datasets.preprocessor import Preprocessor
from modeling.utils import data_util, eval_util, model_util
from modeling.models.ET.dummy_commander import DummyCommander

from teach.inference.actions import obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger

logger = create_logger(__name__)


class ETModel(TeachModel):
    """
    Wrapper around ET Model for inference
    """

    def __init__(self, process_index: int, num_processes: int, model_args: List[str]):
        """Constructor
        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of processes launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1, help="Random seed")
        parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
        parser.add_argument(
            "--commander_model_dir",
            type=str,
            required=True,
            help="Commander model folder name under $TEACH_LOGS")
        parser.add_argument("--driver_model_dir",
                            type=str,
                            required=True,
                            help="Driver model folder name under $TEACH_LOGS")
        parser.add_argument("--checkpoint", type=str, default="latest.pth", help="latest.pth or model_**.pth")
        parser.add_argument("--object_predictor", type=str, required=True, help="Path to MaskRCNN model checkpoint")
        parser.add_argument("--visual_checkpoint", type=str, required=True, help="Path to FasterRCNN model checkpoint")
        parser.add_argument("--preprocessed_data_dir",
                            type=str,
                            required=True,
                            help="preprocessed_data_dir for vocab")
        parser.add_argument("--model_name",
                            type=str,
                            default="ET",
                            help="Name of the agent model.")
        parser.add_argument("--use_dummy_commander", type=bool, default=True, help="")

        args = parser.parse_args(model_args)
        # args.dout = args.model_dir
        self.args = args

        logger.info("ETModel using args %s" % str(args))
        np.random.seed(args.seed)

        self.et_model_args = None
        # self.object_predictor = None
        self.model = None
        self.extractor = None
        self.vocab = None
        self.preprocessor = None

        self.object_predictor = eval_util.load_object_predictor(self.args)

        gpu_count = torch.cuda.device_count()
        logger.info(f"gpu_count: {gpu_count}")
        device = f"cuda:{process_index % gpu_count}" if self.args.device == "cuda" else self.args.device
        self.args.device = device
        logger.info(f"Loading model agent using device: {device}")

        self.driver_model, self.driver_vocab, self.preprocessor, self.driver_model_args = self.set_up_model(process_index, agent="driver")
        self.commander_model, self.commander_vocab, self.preprocessor, self.commander_model_args = self.set_up_model(process_index, agent="commander")
        
        if self.args.use_dummy_commander:
            self.commander_model = DummyCommander()

        self.input_dict = None

    def set_up_model(self, process_index, agent="driver"):
        # os.makedirs(self.args.dout, exist_ok=True)
        if agent == "driver":
            model_dir = self.args.driver_model_dir
        else:
            model_dir = self.args.commander_model_dir

        model_path = os.path.join(model_dir, self.args.checkpoint)
        logger.info(f"Loading {agent} model from {model_path}")

        model_args = model_util.load_model_args(model_path)
        model_args['use_wandb'] = False
        # dataset_info = data_util.read_dataset_info_for_inference(model_dir)
        dataset_info = data_util.read_dataset_info(self.args.preprocessed_data_dir)
        train_data_name = model_args.data["train"][0]

        # train_vocab = data_util.load_vocab_for_inference(model_dir, train_data_name)
        train_vocab = data_util.load_vocab(self.args.preprocessed_data_dir)

        if model_path is not None:
            torch.cuda.empty_cache()
            model, self.extractor = eval_util.load_agent(self.args.model_name, model_path, dataset_info, self.args, test_mode=True)

        vocab = {"word": train_vocab["word"], "action_low": model.vocab_out}
        preprocessor = Preprocessor(vocab=vocab)
        return model, vocab, preprocessor, model_args

    def start_new_tatc_instance(self, tatc_instance):
        self.commander_model.reset()
        self.driver_model.reset()
        self.input_dict = {}
        return True
    
    def featurize_sent(self, sent, agent):
        processed_diag = self.preprocessor.process_sentences([sent])[0]
        processed_diag_toks = self.preprocessor.numericalize(self.driver_vocab["word"],
                            processed_diag,
                            train=False)
        
        agent_tok = self.preprocessor.commander_tok if agent == "commander" else self.preprocessor.driver_tok
        
        # add tokens to denote which agent is speaking
        processed_diag_toks = [agent_tok] + processed_diag_toks + [self.preprocessor.eos_tok]
        return processed_diag_toks

    def featurize_dialogue(self, dialogue):
        dialogue_tokens = []

        for i, (sent, agent) in enumerate(dialogue):
            if not sent:
                continue 

            sent_tokens = self.featurize_sent(sent, agent)
            dialogue_tokens.extend(sent_tokens)
        
        return dialogue_tokens

    def get_next_action_commander(self,
                                  commander_inputs,
                                  tatc_instance,
                                  commander_img_name=None,
                                  tatc_name=None):
        """
        Returns a commander action
        :param commander_inputs: dictionary 
        :param tatc_instance: tatc instance
        :param commander_img_name: commander image file name
        :param tatc_name: tatc instance file name
        """
        # Featurize images
        commander_img_feat = self.extractor.featurize([commander_inputs['commander_img']], batch=1)
        commander_inputs["commander_frames"] = commander_img_feat.unsqueeze(0)

        # dialogue history is a list of strings, agent. need to join them together
        sent_tokens = self.featurize_dialogue(commander_inputs["dialogue_history"])

        # Featurize the dialogue history
        diag_feat = data_util.tensorize_and_pad(
            [(None, {"combined_lang": sent_tokens})], self.args.device, self.driver_model.pad
        )[1]

        commander_inputs.update(diag_feat)

        prev_action = None if len(commander_inputs['commander_action_history']) == 0 else commander_inputs['commander_action_history'][-1]['action']

        with torch.no_grad():
            m_out = self.commander_model.step(
                commander_inputs,
                self.commander_vocab,
                prev_action=prev_action, 
                agent="commander",
                tatc_instance=tatc_instance)

        if not self.args.use_dummy_commander:
            # Predicts commander action and object class
            m_pred = model_util.extract_action_preds(
                m_out,
                self.commander_model.pad,
                self.commander_model.vocab["commander_action_low"],
                clean_special_tokens=False,
                agent="commander")[0]

            action, obj_cls = m_pred["action"], m_pred["obj_cls"]
        else:
            action, obj_cls, utterance = m_out["action"], m_out["obj_cls"], m_out["utterance"]

        logger.debug("Predicted Commander action: %s, obj = %s, utterance = %s" %
                     (str(action), str(obj_cls), utterance))

        action_dict = dict(
            action=action,
            obj_cls=obj_cls,
            utterance=utterance
        )

        return action_dict

    def get_next_action_driver(self,
                               driver_inputs,
                               tatc_instance,
                               driver_img_name=None,
                               tatc_name=None):
        """
        Returns a commander action
        :param driver_inputs: dictionary 
        :param tatc_instance: tatc instance
        :param commander_img_name: commander image file name
        :param tatc_name: tatc instance file name
        """
        driver_img_feat = self.extractor.featurize([driver_inputs['driver_img']], batch=1)
        driver_inputs["driver_frames"] = driver_img_feat

        # Featurize the dialogue history
        sent_tokens = self.featurize_dialogue(driver_inputs["dialogue_history"])

        if not sent_tokens: # this should only happen once at the beginning
            action = 'Text'
            utterance = 'Hello'
            predicted_click = None

            logger.debug("Predicted Driver action: %s, click = %s, utterance = %s" %
                     (str(action), str(predicted_click), utterance))

            # add frame
            frames = driver_inputs["driver_frames"]
            self.driver_model.frames_traj = torch.cat((self.driver_model.frames_traj.to(self.args.device), frames[None]), dim=1)

            action_dict = dict(
                action=action,
                predicted_click=predicted_click,
                utterance=utterance
            )
            return action_dict 
            
        # Featurize the dialogue history
        diag_feat = data_util.tensorize_and_pad(
            [(None, {"combined_lang": sent_tokens})], self.args.device, constants.PAD
        )[1]

        driver_inputs.update(diag_feat)

        prev_action = None if len(driver_inputs['driver_action_history']) == 0 else driver_inputs['driver_action_history'][-1]

        with torch.no_grad():
            m_out = self.driver_model.step(driver_inputs,
                                           self.driver_vocab,
                                           prev_action=prev_action['action'],
                                           agent="driver")

        m_pred = model_util.extract_action_preds(
            m_out,
            self.driver_model.pad,
            self.driver_vocab["action_low"],
            clean_special_tokens=False)[0]

        action = m_pred['action']

        obj = None
        if action in obj_interaction_actions and len(m_pred["object"]) > 0 and len(m_pred["object"][0]) > 0:
            obj = m_pred["object"][0][0]

        predicted_click = None
        if obj is not None:
            predicted_click = self.get_obj_click(obj, driver_inputs['driver_img'])

        # # Simple driver speaker
        # # This can also generated from a text generation language model
        # utterance = None
        if action == "Text":
            import ipdb; ipdb.set_trace()
        #     import ipdb; ipdb.set_trace()
        #     text = "What should I do next" 
        #     utterance = text
        #     utterance = ["<<driver>>"] + self.preprocessor.process_sentences([utterance])[0] + ["<<sent>>"]
        #     utterance = [self.preprocessor.numericalize(self.vocab["word"],
        #                       utterance,
        #                       train=False)
        #                 ]
        #     utterance = torch.tensor(utterance, dtype=torch.long).to(self.args.device)
        #     utterance = self.driver_model.emb_word(utterance)
            
        #     self.input_dict["lang_goal_instr"] = torch.cat([self.input_dict["lang_goal_instr"], utterance], dim=1)

        # Assume previous action succeeded if no better info available
        prev_success = True
        if prev_action is not None and "success" in prev_action:
            prev_success = prev_action["success"]

        utterance = None

        logger.debug("Predicted Driver action: %s, click = %s, utterance = %s" %
                     (str(action), str(predicted_click), utterance))

        # remove blocking actions
        action = self.obstruction_detection(action, prev_success, m_out,
                                            self.driver_model.vocab_out)

        action_dict = dict(
            action=action,
            predicted_click=predicted_click,
            utterance=utterance
        )

        return action_dict

    # def get_next_action(self, img, edh_instance, prev_action, img_name=None, edh_name=None):
    #     """
    #     Sample function producing random actions at every time step. When running model inference, a model should be
    #     called in this function instead.
    #     :param img: PIL Image containing agent's egocentric image
    #     :param edh_instance: EDH instance
    #     :param prev_action: One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values
    #     from a previous call of get_next_action
    #     :param img_name: image file name
    #     :param edh_name: EDH instance file name
    #     :return action: An action name from all_agent_actions
    #     :return obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1) indicating an object in the image;
    #     The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the agent's egocentric image, selects
    #     an object in a 10x10 pixel patch around the pixel indicated by the coordinate if the desired action can be
    #     performed on it, and executes the action in AI2-THOR.
    #     """
    #     img_feat = self.extractor.featurize([img], batch=1)
    #     self.input_dict["frames"] = img_feat

    #     with torch.no_grad():
    #         prev_api_action = None
    #         if prev_action is not None and "action" in prev_action:
    #             prev_api_action = prev_action["action"]
    #         m_out = self.model.step(self.input_dict, self.vocab, prev_action=prev_api_action)

    #     m_pred = model_util.extract_action_preds(
    #         m_out, self.model.pad, self.vocab["action_low"], clean_special_tokens=False
    #     )[0]
    #     action = m_pred["action"]

    #     obj = None
    #     if action in obj_interaction_actions and len(m_pred["object"]) > 0 and len(m_pred["object"][0]) > 0:
    #         obj = m_pred["object"][0][0]

    #     predicted_click = None
    #     if obj is not None:
    #         predicted_click = self.get_obj_click(obj, img)
    #     logger.debug("Predicted action: %s, obj = %s, click = %s" % (str(action), str(obj), str(predicted_click)))

    #     # Assume previous action succeeded if no better info available
    #     prev_success = True
    #     if prev_action is not None and "success" in prev_action:
    #         prev_success = prev_action["success"]

    #     # remove blocking actions
    #     action = self.obstruction_detection(action, prev_success, m_out, self.model.vocab_out)
    #     return action, predicted_click

    def get_obj_click(self, obj_class_idx, img):
        rcnn_pred = self.object_predictor.predict_objects(img)
        obj_class_name = self.object_predictor.vocab_obj.index2word(obj_class_idx)
        candidates = list(filter(lambda p: p.label == obj_class_name, rcnn_pred))
        if len(candidates) == 0:
            return [np.random.uniform(), np.random.uniform()]
        index = np.argmax([p.score for p in candidates])
        mask = candidates[index].mask[0]
        predicted_click = list(np.array(mask.nonzero()).mean(axis=1))
        predicted_click = [
            predicted_click[0] / mask.shape[1],
            predicted_click[1] / mask.shape[0],
        ]
        return predicted_click

    def obstruction_detection(self, action, prev_action_success, m_out, vocab_out):
        """
        change 'MoveAhead' action to a turn in case if it has failed previously
        """
        if action != "Forward" or prev_action_success:
            return action
        dist_action = m_out["action"][0][0].detach().cpu()
        idx_rotateR = vocab_out.word2index("Turn Right")
        idx_rotateL = vocab_out.word2index("Turn Left")
        action = "Turn Left" if dist_action[idx_rotateL] > dist_action[idx_rotateR] else "Turn Right"
        logger.debug("Blocking action is changed to: %s" % str(action))
        return action