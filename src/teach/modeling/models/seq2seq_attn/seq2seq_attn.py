import os
import torch
import numpy as np
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import modeling.models.seq2seq_attn.modules.vnn as vnn
from modeling.models.seq2seq_attn.seq2seq import Module as Base
from modeling.utils import data_util
from modeling.utils.metric_util import compute_f1, compute_exact


class Module(Base):
    def __init__(self, args, embs_ann, vocab, test_mode=False):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # encoder and self-attention
        self.enc = nn.LSTM(args.demb,
                           args.dhid,
                           bidirectional=True,
                           batch_first=True)

        self.enc_att = vnn.SelfAttn(args.dhid * 2)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.progress_aux_loss_wt > 0
                                   or self.args.subgoal_aux_loss_wt > 0)

        decoder = vnn.ConvFrameDecoderProgressMonitor if self.subgoal_monitoring else vnn.ConvFrameDecoder

        if self.args.agent == "driver":
            self.aux_pred_type = "coord"
        elif self.args.agent == "commander":
            self.aux_pred_type = "obj_cls"

        self.num_obj_classes = len(vocab['object_cls'])

        self.dec = decoder(self.emb_action_low,
                           args.dframe,
                           2 * args.dhid,
                           num_obj_classes=self.num_obj_classes,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing,
                           pred=self.aux_pred_type)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = test_mode

        # bce reconstruction loss
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

        self.embs_ann = embs_ann

    def featurize(self, batch, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device(
            'cuda') if self.args.device == "cuda" else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex in batch:
            ###########
            # auxillary
            ###########

            if not self.test_mode:
                # subgoal completion supervision
                # TODO: fix this
                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(
                        np.array(ex['num']['low_to_high_idx']) /
                        self.max_subgoals)

                # progress monitor supervision
                if self.args.progress_aux_loss_wt > 0:
                    num_actions = len(
                        [a for sg in ex['num']['action_low'] for a in sg])
                    subgoal_progress = [(i + 1) / float(num_actions)
                                        for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########
            # goal and instr language
            lang_goal, combined_utts = ex['lang_goal'][0], ex[
                'combined_utterances']

            # zero inputs if specified
            lang_goal = self.zero_input(
                lang_goal) if self.args.zero_goal else lang_goal
            combined_utts = self.zero_input(
                combined_utts) if self.args.zero_instr else combined_utts

            # append goal + instr
            # TODO: fix this, change this to some parameter, should this be min(150, len(combined_utts))?
            max_len = 150
            for t in range(max_len):
                combined_utts_to_t = combined_utts[:t]
                combined_utts_to_t = sum(combined_utts_to_t, [])
                lang_goal_instr = lang_goal + combined_utts_to_t
                feat['lang_goal_instr'].append(lang_goal_instr)

            #########
            # outputs
            #########
            if not self.test_mode:
                # Process commander and driver actions
                feat["commander_action_low"].append([action[0]['action'] for action in ex["actions_low"]])
                feat["driver_action_low"].append([action[1]['action'] for action in ex["actions_low"]])

                # Process additional auxiliary outputs and valid indices
                commander_action_low_aux, driver_action_low_aux, commander_action_low_valid_interact, driver_action_low_valid_interact = [], [], [], []
                for (commander_action, driver_action) in ex['actions_low']:

                    # Add coord when driver successfully interacts with object
                    if driver_action["success"] and "x" in driver_action:
                        driver_action_low_aux.append(
                            [driver_action["x"], driver_action["y"]])
                        driver_action_low_valid_interact.append(1)
                    else:
                        driver_action_low_valid_interact.append(0)

                    # Add object class when commander queries for a given object
                    if commander_action["success"] and commander_action[
                            "action_name"] in ["SearchObject", "SelectOid"]:
                        if commander_action["action_name"] == "SearchObject":
                            obj = commander_action["query"].capitalize()
                        else:
                            obj = commander_action["query"].split(
                                '|')[0].capitalize()

                        # If search object is not in vocabulary, treat it as a failed action
                        if obj not in self.vocab[
                                'object_cls']._word2index.keys():
                            commander_action_low_valid_interact.append(0)
                            continue

                        obj = self.vocab['object_cls'].word2index(obj)
                        commander_action_low_aux.append(obj)
                        commander_action_low_valid_interact.append(1)
                    else:
                        commander_action_low_valid_interact.append(0)

                feat[f"commander_action_low_obj_cls"].append(
                    commander_action_low_aux)
                feat[f"driver_action_low_coord"].append(driver_action_low_aux)
                feat["commander_action_low_valid_interact"].append(
                    commander_action_low_valid_interact)
                feat["driver_action_low_valid_interact"].append(
                    driver_action_low_valid_interact)
        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs,
                                       batch_first=True,
                                       padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq,
                                                    seq_lengths,
                                                    batch_first=True,
                                                    enforce_sorted=False)
                feat[k] = packed_input

            elif k in {'driver_action_low_coord', 'commander_action_low_obj_cls'}:
                seqs = [
                    torch.tensor(vv, device=device, dtype=torch.float)
                    for vv in v
                ]
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [
                    torch.tensor(vv, device=device, dtype=torch.float)
                    for vv in v
                ]
                pad_seq = pad_sequence(seqs,
                                       batch_first=True,
                                       padding_value=self.pad)
                feat[k] = pad_seq
            else:
                # default: tensorize and pad sequence
                seqs = [
                    torch.tensor(vv,
                                 device=device,
                                 dtype=torch.float if
                                 ('frames' in k) else torch.long) for vv in v
                ]
                pad_seq = pad_sequence(seqs,
                                       batch_first=True,
                                       padding_value=self.pad)
                feat[k] = pad_seq
        return feat

    def forward(self, feat, max_decode=300):
        cont_lang, enc_lang = self.encode_lang(feat)
        state_0 = cont_lang[:, 0], torch.zeros_like(cont_lang[:, 0])
        frames = self.vis_dropout(feat['frames'])

        res = self.dec(enc_lang,
                       frames,
                       max_decode=max_decode,
                       gold=feat[f'{self.args.agent}_action_low'],
                       state_0=state_0)
        return res

    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang_goal_instr = feat['lang_goal_instr']
        self.lang_dropout(emb_lang_goal_instr.data)

        enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr)

        if not self.test_mode:
            enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr,
                                                         batch_first=True)

        self.lang_dropout(enc_lang_goal_instr)
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr)

        # cont_lang_goal_instr
        if not self.test_mode:
            cont_lang_goal_instr = cont_lang_goal_instr.view(
                -1, 150, *cont_lang_goal_instr.shape[1:])
            enc_lang_goal_instr = enc_lang_goal_instr.view(
                -1, 150, *enc_lang_goal_instr.shape[1:])

        return cont_lang_goal_instr, enc_lang_goal_instr

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }

    def step(self, feat, vocab, prev_action=None, agent=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features
        if self.r_state['cont_lang'] is None and self.r_state[
                'enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state[
                'enc_lang'] = self.encode_lang(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(
                self.r_state['enc_lang'].size(0), 1)
            self.r_state['state_t'] = self.r_state[
                'cont_lang'], torch.zeros_like(self.r_state['cont_lang'])

        # previous action embedding
        e_t = {}
        if prev_action is not None:
            if agent == "commander":
                e_t["commander"] = self.embed_action([prev_action["commander_action"]], agent="commander").squeeze(0)
            if agent == "driver":
                e_t["driver"] = self.embed_action([prev_action["driver_action"]], agent="driver").squeeze(0)
        else:
            e_t["commander"] = e_t["driver"] = self.r_state['e_t']
        
         # decode and save embedding and hidden states
        
        if agent == "commander":
            with torch.no_grad():    
                out_action_low, out_action_low_aux, state_t, *_ = self.dec.step(
                    self.r_state['enc_lang'],
                    feat['commander_frames'][:, 0],
                    e_t=e_t["commander"],
                    state_tm1=self.r_state['state_t'])
        elif agent == "driver":
            with torch.no_grad(): 
                out_action_low, out_action_low_aux, state_t, *_ = self.dec.step(
                    self.r_state['enc_lang'],
                    feat['driver_frames'][:, 0],
                    e_t=e_t["driver"],
                    state_tm1=self.r_state['state_t'])
        
        # save states
        self.r_state['state_t'] = state_t
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])
        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat[f'out_action_{self.aux_pred_type}'] = out_action_low_aux.unsqueeze(0)
        return feat

    def extract_preds(self, out, batch, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for ex, alow, alow_aux in zip(
                batch, out['out_action_low'].max(2)[1].tolist(),
                out[f'out_action_low_{self.aux_pred_type}']):
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]

            # index to API actions
            words = self.vocab[f'{self.args.agent}_action_low'].index2word(
                alow)

            task_id_ann = self.get_task_and_ann_id(ex)

            pred[task_id_ann] = {
                'action_low': ' '.join(words),
                f'action_{self.aux_pred_type}': alow_aux,
            }

        return pred

    def embed_frames(self, frames_pad):
        """
        take a list of frames tensors, pad it, apply dropout and extract embeddings
        """
        self.vis_dropout(frames_pad)
        frames_4d = frames_pad.view(-1, *frames_pad.shape[2:])
        frames_pad_emb = self.vis_feat(frames_4d).view(*frames_pad.shape[:2],
                                                       -1)
        frames_pad_emb_skip = self.object_feat(frames_4d).view(
            *frames_pad.shape[:2], -1)
        return frames_pad_emb, frames_pad_emb_skip

    def embed_action(self, action, agent="driver"):
        '''
        embed low-level action
        '''
        action_num = torch.tensor(
            self.vocab[f'{agent}_action_low'].word2index(action),
            device=self.args.device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb

    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].view(
            -1, len(self.vocab[f'{self.args.agent}_action_low']))
        l_alow = feat[f'{self.args.agent}_action_low'].view(-1)

        # action loss
        pad_valid = (l_alow != self.pad)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # valid interaction indices
        valid = feat[f'{self.args.agent}_action_low_valid_interact']
        valid_idxs = valid.view(-1).nonzero().view(-1)

        # aux loss
        output_size = 2 if self.aux_pred_type == "coord" else self.num_obj_classes
        flat_p_alow_aux = out[f'out_action_low_{self.aux_pred_type}'].view(
            -1, output_size)[valid_idxs]

        flat_alow_aux = torch.cat(feat[f'{self.args.agent}_action_low_{self.aux_pred_type}'],
                                  dim=0)
        if self.aux_pred_type == "coord":
            loss = self.mse_loss
        elif self.aux_pred_type == "obj_cls":
            loss = self.cross_entropy
            flat_alow_aux = flat_alow_aux.long()

        alow_aux_loss = loss(flat_p_alow_aux, flat_alow_aux).mean()
        losses[
            f'action_low_{self.aux_pred_type}'] = alow_aux_loss * self.args.action_aux_loss_wt

        # TODO: fix this
        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            p_subgoal = feat['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses[
                'subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.progress_aux_loss_wt > 0:
            p_progress = feat['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses[
                'progress_aux'] = self.args.progress_aux_loss_wt * progress_loss

        return losses

    def compute_metric(self, preds, loader):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        data_iter = {key: iter(l) for key, l in loader.items()}

        num_batches = len(next(iter(data_iter.values())))

        for _ in range(num_batches):
            batch = data_util.sample_batches(data_iter, self.args.device,
                                               self.pad, self.args)

            for batch_name, (traj_data, input_dict,
                             gt_dict) in batch.items():

                for task in traj_data:
                    i = self.get_task_and_ann_id(task)

                    if not i in preds: continue 

                    idx = 0 if self.args.agent == "commander" else 1
                    label = ' '.join([
                        a[idx]['action_name']
                        for a in task['actions_low']
                    ])
                    m['action_low_f1'].append(
                        compute_f1(label.lower(), preds[i]['action_low'].lower()))
                    m['action_low_em'].append(
                        compute_exact(label.lower(), preds[i]['action_low'].lower()))

        return {k: sum(v) / len(v) for k, v in m.items()}
