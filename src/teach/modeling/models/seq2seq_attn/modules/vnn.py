import torch
from torch import nn
from torch.nn import functional as F


class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''
    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        scores = F.softmax(self.scorer(inp), dim=1)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''
    def forward(self, inp, h):
        score = self.softmax(inp, h)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2))
        score = F.softmax(raw_score, dim=1)
        return score


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''
    def __init__(self, dframe):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64 * 7 * 7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


# class MaskDecoder(nn.Module):
#     '''
#     mask decoder
#     '''

#     def __init__(self, dhid, pframe=300, hshape=(64,7,7)):
#         super(MaskDecoder, self).__init__()
#         self.dhid = dhid
#         self.hshape = hshape
#         self.pframe = pframe

#         self.d1 = nn.Linear(self.dhid, hshape[0]*hshape[1]*hshape[2])
#         self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.dconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
#         self.dconv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

#     def forward(self, x):
#         x = F.relu(self.d1(x))
#         x = x.view(-1, *self.hshape)

#         x = self.upsample(x)
#         x = self.dconv3(x)
#         x = F.relu(self.bn2(x))

#         x = self.upsample(x)
#         x = self.dconv2(x)
#         x = F.relu(self.bn1(x))

#         x = self.dconv1(x)
#         x = F.interpolate(x, size=(self.pframe, self.pframe), mode='bilinear')

#         return x

class ConvFrameDecoderCommander(ConvFrameDecoder):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

        # Driver image, commander image, driver action, commander action
        self.cell = nn.LSTMCell(dhid + dframe*2 + demb*2, dhid)
        self.aux_pred = nn.Linear(dhid + dhid + dframe*2 + demb*2, self.aux_output_size)
        self.emb_driver = nn.Embedding(len(self.vocab["driver_action_low"]), args.demb)
        self.emb_commander = nn.Embedding(len(self.vocab["commander_action_low"]), args.demb)

    def step(self, inputs, t):
        # previous decoder hidden state
        state_tm1 = inputs['state_t']
        h_tm1 = state_tm1[0]

        # encode visual features
        driver_frame = inputs['driver_frames'][:, t]
        commander_frame = inputs['commander_frames'][:, t]

        # encoder dialogue history
        lang_feat_t = inputs['enc'][:, t]

        driver_vis_feat_t = self.vis_encoder(driver_frame)
        commander_vis_feat_t = self.vis_encoder(commander_frame)

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([commander_vis_feat_t, driver_vis_feat_t, weighted_lang_t, commander_e_t, driver_e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t = state_t[0]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        aux_t = self.aux_pred(cont_t)

        return action_t, aux_t, state_t, lang_attn_t

    def forward(self, inputs, max_decode=150):
        max_t = inputs['gold'].size(1)
        batch = inputs['enc'].size(0)
        driver_e_t = self.go.repeat(batch, 1)
        commander_e_t = self.go.repeat(batch, 1)
        state_t = inputs['state_0']
        inputs['state_t'] = state_t

        actions = []
        attn_scores = []
        aux_output = []

        for t in range(max_t):
            action_t, aux_t, state_t, attn_score_t = self.step(inputs, t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            aux_output.append(aux_t)
            import ipdb; ipdb.set_trace()
            if self.teacher_forcing and self.training:
                # TODO: fix this
                w_t = inputs['gold'][:, t]
            else:
                w_t = action_t.max(1)[1]

            # update action embedding 
            inputs['commander_e_t'] = self.emb_commander(commander_w_t)
            inputs['driver_e_t'] = self.emb_driver(driver_w_t)

            inputs['state_t'] = state_t

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            f'out_action_low_{self.aux_pred_type}': torch.stack(aux_output,
                                                                dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t,
            'out_text': ""
        }
        return results


class ConvFrameDecoder(nn.Module):
    '''
    action decoder
    '''
    def __init__(self,
                 emb,
                 dframe,
                 dhid,
                 num_obj_classes=0,
                 attn_dropout=0.,
                 hstate_dropout=0.,
                 actor_dropout=0.,
                 input_dropout=0.,
                 teacher_forcing=False,
                 pred="coord"):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid + dframe + demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid + dhid + dframe + demb, demb)
        self.aux_pred_type = pred
        self.aux_output_size = 2 if pred == "coord" else num_obj_classes
        self.aux_pred = nn.Linear(dhid + dhid + dframe + demb, self.aux_output_size)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, inputs, t):
        # previous decoder hidden state
        state_tm1 = inputs['state_t']
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        frame = inputs['frames'][:, t]
        lang_feat_t = inputs['enc'][:, t]

        vis_feat_t = self.vis_encoder(frame)

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(
            self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t = state_t[0]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        aux_t = self.aux_pred(cont_t)

        return action_t, aux_t, state_t, lang_attn_t

    def forward(self, inputs, max_decode=150):
        max_t = inputs['gold'].size(1)
        batch = inputs['enc'].size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = inputs['state_0']
        inputs['state_t'] = state_t

        actions = []
        attn_scores = []
        aux_output = []

        for t in range(max_t):
            action_t, aux_t, state_t, attn_score_t = self.step(inputs, t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            aux_output.append(aux_t)
            if self.teacher_forcing and self.training:
                w_t = inputs['gold'][:, t]
            else:
                w_t = action_t.max(1)[1]

            # update action embedding 
            inputs['e_t'] = self.emb(w_t)
            inputs['state_t'] = state_t

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            f'out_action_low_{self.aux_pred_type}': torch.stack(aux_output,
                                                                dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t,
            'out_text': ""
        }
        return results


class ConvFrameCoordDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''
    def __init__(self,
                 emb,
                 dframe,
                 dhid,
                 attn_dropout=0.,
                 hstate_dropout=0.,
                 actor_dropout=0.,
                 input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid + dframe + demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid + dhid + dframe + demb, demb)
        self.coord_pred = nn.Linear(dhid + dhid + dframe + demb, 2)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid + dhid + dframe + demb, 1)
        self.progress = nn.Linear(dhid + dhid + dframe + demb, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc  # language is encoded once at the start

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(
            self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t, c_t = state_t[0], state_t[1]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        coord_t = self.coord_pred(cont_t)

        # predict subgoals completed and task progress
        subgoal_t = F.sigmoid(self.subgoal(cont_t))
        progress_t = F.sigmoid(self.progress(cont_t))

        return action_t, mask_t, state_t, lang_attn_t, subgoal_t, progress_t

    def forward(self, enc, frames, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode,
                                                       frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        coords = []
        attn_scores = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            action_t, coords_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(
                enc, frames[:, t], e_t, state_t)
            coords.append(coords_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_coord': torch.stack(coords, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t': state_t
        }
        return results
