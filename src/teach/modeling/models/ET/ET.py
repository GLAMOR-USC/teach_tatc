import collections
import json
import logging
import os
from importlib import import_module

import wandb
import gtimer as gt
from modeling.utils import data_util, model_util
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from teach.logger import create_logger

logger = create_logger(__name__, level=logging.INFO)

class Module(nn.Module):
    def __init__(self, args, embs_ann, vocab, for_inference=False):
        """
        Abstract model
        """
        nn.Module.__init__(self)
        self.args = args
        self.embs_ann = embs_ann
        self.vocab = vocab
        self.vocab_out = vocab['driver_action_low']
        # sentinel tokens
        self.pad, self.seg = 0, 1
        # summary self.writer
        self.summary_writer = None
        
        # create the model to be trained
        ModelClass = import_module("modeling.models.{}.{}".format(args.model, "transformer")).Model
        self.model = ModelClass(args, embs_ann, self.vocab_out, self.pad, self.seg, for_inference)
        self.for_inference = for_inference

        if self.args.use_wandb:
            wandb.init(
                project="teach_et_baseline", 
                config=self.args,
                name=self.args.name,
                dir=self.args.dout
            )

    def run_train(self, loaders, info, optimizer=None):
        """
        training loop
        """
        # prepare dictionaries
        loaders_train = dict(filter(lambda x: "train" in x[0], loaders.items()))
        assert len(set([len(loader) for loader in loaders_train.values()])) == 1
        vocabs_in = {
            "{};{}".format(loader.dataset.name, loader.dataset.ann_type): loader.dataset.vocab_in
            for loader in loaders.values()
        }
        epoch_length = len(next(iter(loaders_train.values())))
        logger.debug("In LearnedModel.run_train, epoch_length = %d" % epoch_length)

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=self.args.dout)

        # dump config
        with open(os.path.join(self.args.dout, "config.json"), "wt") as f:
            json.dump(vars(self.args), f, indent=2)

        # optimizer
        optimizer, schedulers = model_util.create_optimizer_and_schedulers(
            info["progress"], self.args, self.parameters(), optimizer
        )
        # make sure that all train loaders have the same length
        assert len(set([len(loader) for loader in loaders_train.values()])) == 1
        model_util.save_log(
            self.args.dout,
            progress=info["progress"],
            total=self.args.epochs,
            stage="train",
            best_loss=info["best_loss"],
            iters=info["iters"],
        )

        # display dout
        start_epoch = info['progress'] if self.args.resume else 0 

        logger.info("Saving to: %s" % self.args.dout)
        for epoch in range(start_epoch, self.args.epochs):
            logger.info("Epoch {}/{}".format(epoch, self.args.epochs))
            self.train()
            train_iterators = {key: iter(loader) for key, loader in loaders_train.items()}
            metrics = {key: collections.defaultdict(list) for key in loaders_train}
            gt.reset()

            for _ in tqdm(range(epoch_length), desc="train"):
                # sample batches
                batches = data_util.sample_batches(train_iterators, self.args.device, self.pad, self.args)
                gt.stamp("data fetching", unique=False)

                # do the forward passes
                model_outs, losses_train = {}, {}
                for batch_name, (traj_data, input_dict, gt_dict) in batches.items():
                    if "combined_lang" not in input_dict:
                        raise RuntimeError("In learned.run_train, lang not in input_dict")

                    input_dict['commander_action'] = gt_dict['commander_action']
                    input_dict['driver_action'] = gt_dict['driver_action']

                    model_outs[batch_name] = self.model.forward(vocabs_in[batch_name.split(":")[-1]], **input_dict)
                    info["iters"]["train"] += len(traj_data) if ":" not in batch_name else 0
                gt.stamp("forward pass", unique=False)
                # compute losses
                losses_train = self.model.compute_loss(
                    model_outs,
                    {key: gt_dict for key, (_, _, gt_dict) in batches.items()},
                )

                # do the gradient step
                optimizer.zero_grad()
                sum_loss = sum([sum(loss.values()) for name, loss in losses_train.items()])
                sum_loss.backward()
                optimizer.step()
                gt.stamp("optimizer", unique=False)

                # compute metrics
                for dataset_name in losses_train.keys():
                    self.model.compute_metrics(
                        model_outs[dataset_name],
                        batches[dataset_name][2],
                        metrics["train:" + dataset_name],
                        self.args.compute_train_loss_over_history,
                    )
                    for key, value in losses_train[dataset_name].items():
                        metrics["train:" + dataset_name]["loss/" + key].append(value.item())
                    metrics["train:" + dataset_name]["loss/total"].append(sum_loss.detach().cpu().item())
                gt.stamp("metrics", unique=False)
                if self.args.profile:
                    logger.info(gt.report(include_itrs=False, include_stats=False))

            # save the checkpoint
            logger.info("Saving models...")
            stats = {"epoch": epoch}
            model_util.save_model(self, "model_{:02d}.pth".format(epoch), stats, optimizer=optimizer)
            model_util.save_model(self, "latest.pth", stats, symlink=True)

            # compute metrics for train
            logger.info("Computing train metrics...")
            metrics = {data: {k: sum(v) / len(v) for k, v in metr.items()} for data, metr in metrics.items()}
            stats = {
                "epoch": epoch,
                "general": {"learning_rate": optimizer.param_groups[0]["lr"]},
                **metrics,
            }

            # save the checkpoint
            logger.info("Saving models...")
            model_util.save_model(self, "model_{:02d}.pth".format(epoch), stats, optimizer=optimizer)
            model_util.save_model(self, "latest.pth", stats, symlink=True)
            # write averaged stats
            for loader_id in stats.keys():
                if isinstance(stats[loader_id], dict):
                    for stat_key, stat_value in stats[loader_id].items():
                        # for comparison with old epxs, maybe remove later
                        summary_key = "{}/{}".format(
                            loader_id.replace(":", "/").replace("lmdb/", "").replace(";lang", "").replace(";", "_"),
                            stat_key.replace(":", "/").replace("lmdb/", ""),
                        )

                        if self.args.use_wandb:
                            wandb.log({summary_key: stat_value}, step=info["iters"]["train"])
                        self.summary_writer.add_scalar(summary_key, stat_value, info["iters"]["train"])

            # dump the training info
            model_util.save_log(
                self.args.dout,
                progress=epoch + 1,
                total=self.args.epochs,
                stage="train",
                best_loss=info["best_loss"],
                iters=info["iters"],
            )
            model_util.adjust_lr(self.args, epoch, schedulers)
        logger.info(
            "{} epochs are completed, all the models were saved to: {}".format(self.args.epochs, self.args.dout)
        )

        if self.args.use_wandb:
            wandb.finish()  