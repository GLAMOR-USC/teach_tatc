import argparse
import os
from pathlib import Path
from typing import List
import random
import numpy as np
import torch
from modeling import constants
from modeling.datasets.tatc import TATCDataset
from modeling.datasets.preprocessor import Preprocessor
from modeling.utils import data_util, eval_util, model_util

from teach.inference.actions import obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger

logger = create_logger(__name__)


class Seq2SeqModel(TeachModel):
    """
    Wrapper around Seq2Seq Model for teach inference
    """
    def __init__(self, process_index: int, num_processes: int,
                 model_args: List[str]):
        """Constructor

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of pro∏cesses launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1, help="Random seed")
        parser.add_argument("--device",
                            type=str,
                            default="cuda",
                            help="cpu or cuda")
        parser.add_argument(
            "--commander_model_dir",
            type=str,
            required=True,
            help="Commander model folder name under $TEACH_LOGS")
        parser.add_argument("--driver_model_dir",
                            type=str,
                            required=True,
                            help="Driver model folder name under $TEACH_LOGS")
        parser.add_argument("--checkpoint",
                            type=str,
                            default="latest.pth",
                            help="latest.pth or model_**.pth")
        parser.add_argument("--visual_checkpoint",
                            type=str,
                            required=True,
                            help="Path to FasterRCNN model checkpoint")
        parser.add_argument("--preprocessed_data_dir",
                            type=str,
                            required=True,
                            help="preprocessed_data_dir for vocab")
        parser.add_argument("--model_name",
                            type=str,
                            default="seq2seq_attn",
                            help="Name of the agent model.")

        args = parser.parse_args(model_args)
        args.dout_commander = args.commander_model_dir
        args.dout_driver = args.driver_model_dir

        self.args = args

        logger.info("Seq2SeqModel using args %s" % str(args))
        np.random.seed(args.seed)

        self.model_args = None
        self.extractor = None
        self.vocab = None

        # Setup commander and driver models
        self.commander_model = self.set_up_model(process_index, agent="commander")
        self.driver_model = self.set_up_model(process_index, agent="driver")
        self.vocab = self.commander_model.vocab
        self.preprocessor = Preprocessor(vocab=self.driver_model.vocab)

        self.input_dict = None
        self.cur_tatc_instance = None
        self.pc_result = None

    def set_up_model(self, process_index, agent="driver"):
        if agent == "commander":
            model_path = os.path.join(self.args.commander_model_dir,
                                      self.args.checkpoint)
            os.makedirs(self.args.dout_commander, exist_ok=True)
        else:
            model_path = os.path.join(self.args.driver_model_dir,
                                      self.args.checkpoint)
            os.makedirs(self.args.dout_driver, exist_ok=True)

        logger.info(f"Loading {agent} model from {model_path}")

        model_args = model_util.load_model_args(model_path)
        dataset_info = data_util.read_dataset_info(self.args.preprocessed_data_dir)

        train_data_name = model_args.data["train"][0]
        train_vocab = data_util.load_vocab(self.args.preprocessed_data_dir)

        # Load model from checkpoint
        if model_path is not None:
            torch.cuda.empty_cache()
            gpu_count = torch.cuda.device_count()
            logger.info(f"gpu_count: {gpu_count}")
            device = f"cuda:{process_index % gpu_count}" if self.args.device == "cuda" else self.args.device
            self.args.device = device
            logger.info(f"Loading {agent} model agent using device: {device}")
            model, self.extractor = eval_util.load_agent(self.args.model_name,
                                                         model_path,
                                                         dataset_info,
                                                         self.args,
                                                         test_mode=True)
        return model

    def start_new_tatc_instance(self, tatc_instance):
        self.commander_model.reset()
        self.driver_model.reset()

        # Setup input for tatc instance and process the goal instruction
        # We don't need to process the goal instruction
        self.input_dict = {}
        # tatc_instance = self.preprocessor.process_goal_instr(
        #     tatc_instance, is_test_split=True)
        # lang_goal = torch.tensor(tatc_instance["lang_goal"],
        #                          dtype=torch.long).to(self.args.device)

        # # Embed the language goal
        # lang_goal = self.commander_model.emb_word(lang_goal)
        # self.input_dict["lang_goal_instr"] = lang_goal
        return True

    def extract_progress_check_subtask_string(self, pc_result, task=None):
        if pc_result["success"]:
            return ""
        
        for subgoal in pc_result["subgoals"]:
            if subgoal["success"] == 1:
                continue 
            
            if subgoal["steps"][0]["success"] == 0 and len(subgoal["description"])>0:
                return subgoal["description"]
            
            for step in subgoal["steps"]:
                if step["success"] == 1: 
                    continue 
                
                if len(step["desc"])>0:
                    return step["desc"]
    
        return task
    
    def featurize_sent(self, sent, agent):
        processed_diag = self.preprocessor.process_sentences([sent])[0]
        processed_diag_toks = self.preprocessor.numericalize(self.vocab["word"],
                            processed_diag,
                            train=False)
        
        agent_tok = self.preprocessor.commander_tok if agent == "commander" else self.preprocessor.driver_tok
        processed_diag_toks = [agent_tok] + processed_diag_toks + [self.preprocessor.eos_tok]
        
        sent_emb = self.commander_model.emb_word(torch.Tensor(processed_diag_toks).to(self.args.device).long())
        return sent_emb

    def featurize_dialogue(self, dialogue):
        dialogue_emb = None

        for i, (sent, agent) in enumerate(dialogue):
            sent_emb = self.featurize_sent(sent, agent)

            if dialogue_emb is None:
                dialogue_emb = sent_emb 
            else:
                dialogue_emb = torch.cat([dialogue_emb, sent_emb], dim=0)

        # add a batch dimension
        return dialogue_emb.unsqueeze(0)

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

        self.input_dict["commander_frames"] = commander_img_feat.unsqueeze(0)

        # Featurize the dialogue history
        self.input_dict['dialogue_hist'] = self.featurize_dialogue(commander_inputs["dialogue_history"])

        prev_action = None if len(commander_inputs['commander_action_history']) == 0 else commander_inputs['commander_action_history'][-1]['action']

        import ipdb; ipdb.set_trace()
        with torch.no_grad():
            m_out = self.commander_model.step(
                self.input_dict,
                self.vocab,
                prev_action=prev_action, 
                agent="commander")

        # Predicts commander action and object class
        m_pred = model_util.extract_action_preds(
            m_out,
            self.commander_model.pad,
            self.commander_model.vocab["commander_action_low"],
            clean_special_tokens=False,
            agent="commander")[0]

        action, obj_cls = m_pred["action"], m_pred["obj_cls"]

        # Simple logic to force Commander to check progress and produce utterance
        if not prev_action or (prev_action != None and prev_action == action):
            print('Forcing progress check!')
            action = 'OpenProgressCheck'

        if prev_action == "OpenProgressCheck":
            action = "Text"

        # Simple commander speaker that uses the text from progress check output
        utterance = None
        if action == "Text":
                
            # Get pc results
            pc_results = commander_inputs['commander_action_result_history'][-1]

            task_desc = tatc_instance['tasks'][0]['desc']
            next_instr = self.extract_progress_check_subtask_string(pc_results, task=tatc_instance['tasks'][0]['desc'])


            # Check if goal already provided 
            goal_provided = False 
            for sent, _ in commander_inputs['dialogue_history']:
                if task_desc == sent:
                    goal_provided = True 

            instr = task_desc if not goal_provided else next_instr
            utterance = instr
            # utterance = self.featurize_sent(instr, "commander")                
            
        # Assume previous action succeeded if no better info available
        prev_success = True
        if prev_action is not None and "success" in prev_action:
            prev_success = prev_action["success"]

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
                               prev_action,
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
        self.input_dict["driver_frames"] = driver_img_feat.unsqueeze(0)

        # Featurize the dialogue history
        if not driver_inputs["dialogue_history"]:
            driver_inputs["dialogue_history"] = [("", "driver")]
        
        self.input_dict['dialogue_hist'] = self.featurize_dialogue(driver_inputs["dialogue_history"])

        prev_action = None if len(driver_inputs['driver_action_history']) == 0 else driver_inputs['driver_action_history'][-1]['action']

        import ipdb; ipdb.set_trace()
        with torch.no_grad():
            m_out = self.driver_model.step(self.input_dict,
                                           self.vocab,
                                           prev_action=prev_action,
                                           agent="driver")

        m_pred = model_util.extract_action_preds(
            m_out,
            self.driver_model.pad,
            self.driver_model.vocab["driver_action_low"],
            clean_special_tokens=False,
            agent="driver")[0]

        action, obj_relative_coord = m_pred["action"], m_pred['coord']

        if not action in obj_interaction_actions:
            obj_relative_coord = None

        # if prev_action!=None and prev_action['driver_action']==action:
        #     # choosing random navigation action
        #     action = random.choice(['Forward', 'Backward', 'Turn Left', 'Turn Right'])
       
        # Simple driver speaker
        # This can also generated from a text generation language model
        utterance = None
        if action == "Text":
            import ipdb; ipdb.set_trace()
            text = "What should I do next" 
            utterance = text
            utterance = ["<<driver>>"] + self.preprocessor.process_sentences([utterance])[0] + ["<<sent>>"]
            utterance = [self.preprocessor.numericalize(self.vocab["word"],
                              utterance,
                              train=False)
                        ]
            utterance = torch.tensor(utterance, dtype=torch.long).to(self.args.device)
            utterance = self.driver_model.emb_word(utterance)
            
            self.input_dict["lang_goal_instr"] = torch.cat([self.input_dict["lang_goal_instr"], utterance], dim=1)

        # Assume previous action succeeded if no better info available
        prev_success = True
        if prev_action is not None and "success" in prev_action:
            prev_success = prev_action["success"]

        logger.debug("Predicted Driver action: %s, click = %s, utterance = %s" %
                     (str(action), str(obj_relative_coord), utterance))

        # remove blocking actions
        action = self.obstruction_detection(action, prev_success, m_out,
                                            self.driver_model.vocab_out)

        action_dict = dict(
            action=action,
            obj_relative_coord=obj_relative_coord,
            utterance=utterance
        )

        return action_dict

    def get_obj_click(self, obj_class_idx, img):
        rcnn_pred = self.object_predictor.predict_objects(img)
        obj_class_name = self.object_predictor.vocab_obj.index2word(
            obj_class_idx)
        candidates = list(
            filter(lambda p: p.label == obj_class_name, rcnn_pred))
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

    def obstruction_detection(self, action, prev_action_success, m_out,
                              vocab_out):
        """
        change 'MoveAhead' action to a turn in case if it has failed previously
        """
        if action != "Forward" or prev_action_success:
            return action
        dist_action = m_out["action"][0][0].detach().cpu()
        idx_rotateR = vocab_out.word2index("Turn Right")
        idx_rotateL = vocab_out.word2index("Turn Left")
        action = "Turn Left" if dist_action[idx_rotateL] > dist_action[
            idx_rotateR] else "Turn Right"
        logger.debug("Blocking action is changed to: %s" % str(action))
        return action
