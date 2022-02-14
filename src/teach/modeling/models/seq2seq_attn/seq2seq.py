import os
import random
import json
import torch
import pprint
from tqdm import tqdm
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange
import logging
from teach.logger import create_logger
from modeling.utils import data_util

logger = create_logger(__name__, level=logging.DEBUG)


class Module(nn.Module):
    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # sentinel tokens
        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)

        # action embedding specific to driver and commander
        action_emb_key = "driver_action_low" if self.args.agent == "driver" else "commander_action_low"
        self.emb_action_low = nn.Embedding(len(vocab[action_emb_key]),
                                           args.demb)
        self.vocab_out = vocab[action_emb_key]

        # end tokens
        self.stop_token = self.vocab['word'].word2index("<<stop>>",
                                                        train=False)
        self.seg_token = self.vocab['word'].word2index("<<seg>>", train=False)

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)

        # summary self.writer
        self.summary_writer = None

    def run_train(self, loaders, info, args=None, optimizer=None):
        '''
        training loop
        '''
        # args
        args = args or self.args

        # setup data loaders
        loaders_train = dict(filter(lambda x: "train" in x[0],
                                    loaders.items()))
        loaders_valid_seen = dict(
            filter(lambda x: "valid_seen" in x[0], loaders.items()))
        loaders_valid_unseen = dict(
            filter(lambda x: "valid_unseen" in x[0], loaders.items()))

        assert len(set([len(loader)
                        for loader in loaders_train.values()])) == 1

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(),
                                                  lr=args.lr['init'])

        # display dout
        print("Saving to: %s" % self.args.dout)

        best_loss = {'train': 1e10, 'valid_seen': 1e10, 'valid_unseen': 1e10}
        train_iter, valid_seen_iter, valid_unseen_iter = 0, 0, 0

        logger.info("Saving to: %s" % args.dout)

        # training loop
        for epoch in range(info["progress"], args.epochs):
            p_train, train_iter, total_train_loss, m_train = self.train_one_epoch(loaders_train, epoch, optimizer, args=args, name='train', iteration=train_iter)

            # compute metrics for valid_seen
            p_valid_seen, valid_seen_iter, total_valid_seen_loss, m_valid_seen = self.run_pred(
                loaders_valid_seen,
                args=args,
                name='valid_seen',
                iteration=valid_seen_iter)

            m_valid_seen.update(self.compute_metric(p_valid_seen,
                                                    loaders_valid_seen))
            m_valid_seen['total_loss'] = float(total_valid_seen_loss)
            self.summary_writer.add_scalar('valid_seen/total_loss',
                                           m_valid_seen['total_loss'],
                                           valid_seen_iter)

            # compute metrics for valid_unseen
            p_valid_unseen, valid_unseen_iter, total_valid_unseen_loss, m_valid_unseen = self.run_pred(
                loaders_valid_unseen,
                args=args,
                name='valid_unseen',
                iteration=valid_unseen_iter)

            m_valid_unseen.update(
                self.compute_metric(p_valid_unseen, loaders_valid_unseen))
            m_valid_unseen['total_loss'] = float(total_valid_unseen_loss)
            self.summary_writer.add_scalar('valid_unseen/total_loss',
                                           m_valid_unseen['total_loss'],
                                           valid_unseen_iter)

            stats = {
                'epoch': epoch,
                'valid_seen': m_valid_seen,
                'valid_unseen': m_valid_unseen
            }

            # new best valid_seen loss
            if total_valid_seen_loss < best_loss['valid_seen']:
                print('\nFound new best valid_seen!! Saving...')
                fsave = os.path.join(args.dout, 'best_seen.pth')
                torch.save(
                    {
                        'metric': stats,
                        'model': self.state_dict(),
                        'optim': optimizer.state_dict(),
                        'args': self.args,
                        'embs_ann': self.embs_ann,
                        'vocab': self.vocab,
                    }, fsave)
                fbest = os.path.join(args.dout, 'best_seen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                fpred = os.path.join(args.dout, 'valid_seen.debug.preds.json')
                with open(fpred, 'wt') as f:
                    json.dump(self.make_debug(p_valid_seen, loaders_valid_seen),
                              f,
                              indent=2)
                best_loss['valid_seen'] = total_valid_seen_loss

            # new best valid_unseen loss
            if total_valid_unseen_loss < best_loss['valid_unseen']:
                print('Found new best valid_unseen!! Saving...')
                fsave = os.path.join(args.dout, 'best_unseen.pth')
                torch.save(
                    {
                        'metric': stats,
                        'model': self.state_dict(),
                        'optim': optimizer.state_dict(),
                        'args': self.args,
                        'embs_ann': self.embs_ann,
                        'vocab': self.vocab,
                    }, fsave)
                fbest = os.path.join(args.dout, 'best_unseen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                fpred = os.path.join(args.dout,
                                     'valid_unseen.debug.preds.json')
                with open(fpred, 'wt') as f:
                    json.dump(self.make_debug(p_valid_unseen, loaders_valid_unseen),
                              f,
                              indent=2)

                best_loss['valid_unseen'] = total_valid_unseen_loss

            # save the latest checkpoint
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')

            torch.save(
                {
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)

            # debug action output json for train
            fpred = os.path.join(args.dout, 'train.debug.preds.json')
            with open(fpred, 'wt') as f:
                json.dump(self.make_debug(p_train, loaders_train), f, indent=2)

            # write stats
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v,
                                                       train_iter)
            pprint.pprint(stats)

    def train_one_epoch(self, loaders_train, epoch, optimizer, args=None, name='train', iteration=0):
        logger.info("Epoch {}/{}".format(epoch, args.epochs))
        self.train()

        train_iterators = {
            key: iter(loader)
            for key, loader in loaders_train.items()
        }

        p_train = {}
        m_train = collections.defaultdict(list)
        total_train_loss = list()
        train_iter = iteration

        epoch_length = len(next(iter(loaders_train.values())))

        for _ in tqdm(range(epoch_length), desc="train"):
        # for _ in tqdm(range(2), desc="train"):
            # sample batches
            batches = data_util.sample_batches(train_iterators,
                                                self.args.device, self.pad,
                                                self.args)

            # iterate over batches
            for batch_name, (traj_data, input_dict,
                                gt_dict) in batches.items():
                feat = self.featurize(traj_data)
                feat['frames'] = input_dict['frames']

                # Compute forward pass of model
                m_out = self.forward(feat)

                # Given the model output, convert into executable action
                m_preds = self.extract_preds(m_out, traj_data)
                p_train.update(m_preds)

                loss = self.compute_loss(m_out, traj_data, feat)

                for k, v in loss.items():
                    ln = 'loss_' + k
                    m_train[ln].append(v.item())
                    self.summary_writer.add_scalar('train/' + ln, v.item(),
                                                    train_iter)

                # optimizer backward pass
                optimizer.zero_grad()
                sum_loss = sum(loss.values())
                sum_loss.backward()
                optimizer.step()

                self.summary_writer.add_scalar('train/loss', sum_loss,
                                                train_iter)

                sum_loss = sum_loss.detach().cpu()
                total_train_loss.append(float(sum_loss))
                train_iter += self.args.batch

                del feat, m_out, loss, m_preds
                torch.cuda.empty_cache()

            del batches 
            torch.cuda.empty_cache()
        
        return p_train, train_iter, total_train_loss, m_train


    def run_pred(self, dev, args=None, name='dev', iteration=0):
        '''
        validation loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()

        total_loss = list()
        dev_iter = iteration

        data_iter = {key: iter(loader) for key, loader in dev.items()}
        
        num_batches = len(next(iter(data_iter.values())))
        
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc=name):
            # for _ in tqdm(range(1), desc=name):
                # sample batches
                batches = data_util.sample_batches(data_iter, self.args.device,
                                                self.pad, self.args)

                for batch_name, (traj_data, input_dict,
                                gt_dict) in batches.items():
                    feat = self.featurize(traj_data, load_frames=False)
                    feat['frames'] = input_dict['frames']

                    m_out = self.forward(feat)
                    m_preds = self.extract_preds(m_out, traj_data)
                    p_dev.update(m_preds)
                    loss = self.compute_loss(m_out, traj_data, feat)

                    for k, v in loss.items():
                        ln = 'loss_' + k
                        m_dev[ln].append(v.item())
                        self.summary_writer.add_scalar("%s/%s" % (name, ln),
                                                    v.item(), dev_iter)
                    sum_loss = sum(loss.values())
                    self.summary_writer.add_scalar("%s/loss" % (name), sum_loss,
                                                dev_iter)
                    total_loss.append(float(sum_loss.detach().cpu()))
                    dev_iter += len(traj_data)

                    del feat, m_out, loss, m_preds
                    torch.cuda.empty_cache()

                del batches 
                torch.cuda.empty_cache()

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev

    def featurize(self, batch):
        raise NotImplementedError()

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['tasks'][0]['task_id'], str(ex['repeat_idx']))

    def make_debug(self, preds, loader):
        '''
        readable output generator for debugging
        '''
        debug = {}
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
                    debug[i] = {
                        'lang_goal': task['tasks'][0]['task_name'],
                        'action_low': [
                            a[idx]['action_name']
                            for a in task['actions_low']
                        ],
                        'p_action_low':
                        preds[i]['action_low'].split(),
                    }
        return debug

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.data, task['task'],
                                 '%s' % self.args.pp_folder,
                                 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'],
                            *(ex['root'].split('/')[-2:]))

    def zero_input(self, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1**(epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
