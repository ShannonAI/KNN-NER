#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : build_datastore.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/07 16:40
@version: 1.0
@desc  :
"""

import os
import json
import argparse
from functools import partial

from datasets.collate_functions import collate_to_max_length
from datasets.ner_dataset import NERDataset
from ner_trainer import NERTask
from utils.random_seed import set_random_seed

# enable reproducibility
# https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
set_random_seed(2333)

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from transformers import BertConfig

import pytorch_lightning as pl
from pytorch_lightning import Trainer


class Datastore(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args

        self.entity_labels = NERDataset.get_labels(os.path.join(args.data_dir, "ner_labels.txt"))
        self.bert_dir = args.bert_path
        self.num_labels = len(self.entity_labels)
        self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=True,
                                                      return_dict=True, num_labels=self.num_labels)
        self.model = None
    def forward(self, input_ids, pinyin_ids):
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, pinyin_ids, attention_mask=attention_mask)

    def test_dataloader(self, ) -> DataLoader:
        dataset = NERDataset(directory=self.args.data_dir, prefix="train",
                             vocab_file=os.path.join(self.args.bert_path, "vocab.txt"),
                             max_length=self.args.max_length,
                             config_path=os.path.join(self.args.bert_path, "config"),
                             file_name=self.args.file_name)

        batch_size = self.args.batch_size
        data_sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                                drop_last=False)

        return dataloader

    def test_step(self, batch, batch_idx):
        input_ids, pinyin_ids, gold_labels = batch
        sequence_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)
        model_output = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids,)    # [bsz, sent_len, feature_size]
        # [bsz, sent_len, feature_size], [bsz, sent_len], [bsz, sent_len]
        return {"features": model_output.hidden_states[-1], "labels": gold_labels, "mask": sequence_mask}

    def test_epoch_end(self, outputs):
        hidden_size = outputs[0]['features'].shape[2]
        token_sum = sum(int(x['mask'].sum(dim=-1).sum(dim=-1).cpu()) for x in outputs)

        data_store_key_in_memory = np.zeros((token_sum, hidden_size), dtype=np.float32)
        data_store_val_in_memory = np.zeros((token_sum,), dtype=np.int32)

        now_cnt = 0
        for x in outputs:
            features = x['features'].reshape(-1, hidden_size)
            mask = x['mask'].bool()
            labels = torch.masked_select(x['labels'], mask).cpu().numpy()
            mask = mask.reshape(features.shape[0], 1).expand(features.shape[0], features.shape[1])
            features = torch.masked_select(features, mask).view(-1, hidden_size).cpu()
            np_features = features.numpy().astype(np.float32)
            data_store_key_in_memory[now_cnt:now_cnt+np_features.shape[0]] = np_features
            data_store_val_in_memory[now_cnt:now_cnt+np_features.shape[0]] = labels
            now_cnt += np_features.shape[0]

        datastore_info = {
            "token_sum": token_sum,
            "hidden_size": hidden_size
        }
        json.dump(datastore_info, open(os.path.join(self.args.datastore_path, "datastore_info.json"), "w"),
                    sort_keys=True, indent=4, ensure_ascii=False)

        key_file = os.path.join(self.args.datastore_path, "keys.npy")
        keys = np.memmap(key_file, 
                     dtype=np.float32,
                     mode="w+",
                     shape=(token_sum, hidden_size))

        val_file = os.path.join(self.args.datastore_path, "vals.npy")
        vals = np.memmap(val_file, 
                     dtype=np.int32,
                     mode="w+",
                     shape=(token_sum,))
        
        keys[:] = data_store_key_in_memory[:]
        vals[:] = data_store_val_in_memory[:]

        return {"saved dir": self.args.datastore_path}


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--file_name", default="", type=str, help="use for truncated sets.")
    parser.add_argument("--path_to_model_hparams_file", default="", type=str, help="use for evaluation")
    parser.add_argument("--checkpoint_path", default="", type=str, help="use for evaluation.")
    parser.add_argument("--datastore_path", default="", type=str, help="use for saving datastore.")

    return parser

def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ner_model = NERTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                            hparams_file=args.path_to_model_hparams_file,
                                                            map_location=None,
                                                            batch_size=args.batch_size)
    model = Datastore(args)
    model.model = ner_model.model
    trainer = Trainer.from_argparse_args(args, deterministic=True)

    trainer.test(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()