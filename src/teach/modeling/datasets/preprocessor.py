import os
import copy

import json
import revtok

import modeling.constants as constants
from modeling.utils import data_util
from vocab import Vocab


class Preprocessor(object):
    def __init__(self,
                 vocab,
                 subgoal_ann=False,
                 is_test_split=False,
                 frame_size=300):
        self.subgoal_ann = subgoal_ann
        self.is_test_split = is_test_split
        self.frame_size = frame_size

        if vocab is None:
            self.vocab = {
                "word":
                Vocab(["<<pad>>", "<<seg>>", "<<goal>>", "<<mask>>"]),
                "action_low":
                Vocab(["<<pad>>", "<<seg>>", "<<stop>>", "<<mask>>"]),
                "action_high":
                Vocab(["<<pad>>", "<<seg>>", "<<stop>>", "<<mask>>"]),
            }
        else:
            self.vocab = vocab

        self.word_seg = self.vocab["word"].word2index("<<seg>>", train=False)
        self.commander_tok = self.vocab["word"].word2index("<<commander>>",
                                                           train=True)
        self.driver_tok = self.vocab["word"].word2index("<<driver>>",
                                                        train=True)

    @staticmethod
    def numericalize(vocab, words, train=True):
        """
        Converts words to unique integers
        """
        if not train:
            new_words = set(words) - set(vocab.counts.keys())
            if new_words:
                # replace unknown words with <<pad>>
                words = [w if w not in new_words else "<<pad>>" for w in words]
        return vocab.word2index(words, train=train)

    def process_sentences(self, sentences):
        """
        Text preprocessing for sentences. Split sentence into tokens, remove spaces, and lowercase.
        """
        sentences = [
            revtok.tokenize(data_util.remove_spaces_and_lower(sent))
            for sent in sentences
        ]
        sentences = [[w.strip().lower() for w in sent] for sent in sentences]
        return sentences

    def process_goal_instr(self, traj, is_test_split=False):
        """
        Process high-level goal instruction
        """
        if self.is_test_split:
            is_test_split = True

        goal_desc = traj["tasks"][0]["desc"]
        processed_goal_instr = self.process_sentences([goal_desc])[0]
        traj["lang_goal"] = [
            self.numericalize(self.vocab["word"],
                              processed_goal_instr,
                              train=not is_test_split)
        ]
        return traj

    def process_language(self, ex, traj, is_test_split=False):
        """
        Process commander + driver dialogues, commander progress check outputs, etc. Add words to vocabulary.
        """
        if self.is_test_split:
            is_test_split = True

        # Process agent dialogues
        commander_utterances, driver_utterances = [], []
        interactions = traj["tasks"][0]["episodes"][0]["interactions"]

        # Time align agent dialogue
        # Either commander or driver speaks at each time step and other agent produces an empty string
        for interaction in interactions:
            if "utterance" in interaction:
                if interaction["agent_id"] == 0:  # Commander
                    commander_utterances.append(interaction["utterance"])
                    driver_utterances.append("")
                elif interaction["agent_id"] == 1:  # Driver
                    driver_utterances.append(interaction["utterance"])
                    commander_utterances.append("")
            else:
                commander_utterances.append("")
                driver_utterances.append("")

        # Process goal instruction
        traj = self.process_goal_instr(traj, is_test_split)
        c_toks = self.process_sentences(commander_utterances)
        d_toks = self.process_sentences(driver_utterances)

        commander_utts_tok, driver_utts_tok, combined_utts_tok = [], [], []

        for (commander_utt, driver_utt) in zip(c_toks, d_toks):
            # Add an end sentence token if the utterance_t isn't an empty sentence
            # Also add a token to mark which agent is speaking
            if commander_utt:
                commander_utts_tok.append(["<<commander>>"] + commander_utt +
                                          ["<<sent>>"])
                combined_utts_tok.append(["<<commander>>"] + commander_utt +
                                         ["<<sent>>"])
            else:
                commander_utts_tok.append([])

            if driver_utt:
                driver_utts_tok.append(["<<driver>>"] + driver_utt +
                                       ["<<sent>>"])
                combined_utts_tok.append(["<<driver>>"] + driver_utt +
                                         ["<<sent>>"])
            else:
                driver_utts_tok.append([])

            if not commander_utt and not driver_utt:
                combined_utts_tok.append([])


        commander_utts_tok.append(["<<stop>>"])
        driver_utts_tok.append(["<<stop>>"])
        combined_utts_tok.append(["<<stop>>"])

        # Save tokens to traj info
        traj["commander_utterances_tok"] = commander_utts_tok
        traj["driver_utterances_tok"] = driver_utts_tok
        traj["combined_utts_tok"] = combined_utts_tok

        # Convert each sentence into numerical tokens
        traj["commander_utterances"] = [
            self.numericalize(self.vocab["word"], x, train=not is_test_split)
            for x in commander_utts_tok
        ]

        traj["driver_utterances"] = [
            self.numericalize(self.vocab["word"], x, train=not is_test_split)
            for x in driver_utts_tok
        ]

        traj["combined_utterances"] = [
            self.numericalize(self.vocab["word"], x, train=not is_test_split)
            for x in combined_utts_tok
        ]

        # Process progress check vocabulary for commander
        for (commander_action, _) in traj["actions_low"]:
            if commander_action["action_name"] == "OpenProgressCheck":
                if not "pc_json" in commander_action:
                    print('Progress check not found...')
                    # import ipdb
                    # ipdb.set_trace()
                    continue

                pc_json_f = commander_action["pc_json"]

                # Concatenate all the high-level and low-level instructions into a single string
                all_words = ""

                with open(os.path.join(constants.TEACH_DATA, pc_json_f)) as f:
                    pc_output = json.load(f)

                    all_words += pc_output["task_desc"] + " "
                    subgoals = pc_output["subgoals"]

                    for subgoal in subgoals:
                        all_words += subgoal["description"] + " "

                        for steps in subgoal["steps"]:
                            all_words += steps["desc"] + " "

                # Process the concatenated string and add words to vocabulary
                all_words = revtok.tokenize(
                    data_util.remove_spaces_and_lower(all_words))
                all_words = [w.strip().lower() for w in all_words]
                [
                    self.vocab["word"].word2index(w, train=not is_test_split)
                    for w in all_words
                ]

    def process_actions(self, ex, traj, is_test_split=False):
        # Action at each timestep is a tuple of (Commander Action, Driver Action)
        traj["actions_low"] = list()

        idx_to_action_json = "meta_data_files/ai2thor_resources/action_idx_to_action_name.json"
        action_to_idx_json = "meta_data_files/ai2thor_resources/action_to_action_idx.json"

        with open(os.path.join(constants.TEACH_SRC, idx_to_action_json)) as f:
            idx_to_action_name = json.load(f)

        with open(os.path.join(constants.TEACH_SRC, action_to_idx_json)) as f:
            action_to_idx = json.load(f)

        all_interactions = ex['tasks'][0]['episodes'][0]['interactions']

        # Create no op actions for the driver and commander
        no_op_commander = dict(
            agent_id=0,
            action=self.vocab["commander_action_low"].word2index(
                "NoOp", train=not is_test_split),
            action_name="NoOp",
            success=1,
            query="",
            commander_obs="",
            driver_obs="",
            duration=1)

        no_op_driver = no_op_commander.copy()
        no_op_driver["action"] = self.vocab["driver_action_low"].word2index(
            "NoOp", train=not is_test_split)
        no_op_driver['agent_id'] = 1

        # For each interaction, add the action id and action name
        for i, action in enumerate(all_interactions):
            action_dict = action.copy()
            idx = action["action_id"]
            action_idx = action_to_idx[str(idx)]  # get the actual index
            action_name = action_dict["action_name"] = idx_to_action_name[str(
                action_idx)]  # get the action name

            key = "driver_action_low" if action_dict[
                "agent_id"] == 1 else "commander_action_low"

            action_dict["action"] = self.vocab[key].word2index(
                action_name, train=not is_test_split)

            if action_dict["agent_id"] == 0:  # Commander
                no_op_driver['time_start'] = action_dict['time_start']
                traj["actions_low"].append([action_dict, no_op_driver])
            else:  # Driver
                no_op_commander['time_start'] = action_dict['time_start']
                traj["actions_low"].append([no_op_commander, action_dict])