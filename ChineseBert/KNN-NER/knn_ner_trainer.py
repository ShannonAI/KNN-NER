#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : knn_ner_trainer.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/07 23:50
@version: 1.0
@desc  :
"""

import os
import json
import argparse
import logging
from functools import partial

from datasets.collate_functions import collate_to_max_length
from datasets.ner_dataset import NERDataset
from utils.random_seed import set_random_seed
from metrics.ner import SpanF1ForNER
from ner_trainer import NERTask

# enable reproducibility
# https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
set_random_seed(2333)

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from transformers import BertConfig

import pytorch_lightning as pl
from pytorch_lightning import Trainer


class KNNNERTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args

        self.entity_labels = NERDataset.get_labels(os.path.join(args.data_dir, "ner_labels.txt"))
        self.bert_dir = args.bert_path
        self.num_labels = len(self.entity_labels)
        self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=True,
                                                      num_labels=self.num_labels,
                                                      hidden_dropout_prob=self.args.hidden_dropout_prob)
        self.model = None

        self.ner_evaluation_metric = SpanF1ForNER(entity_labels=self.entity_labels, save_prediction=self.args.save_ner_prediction)

        self.num_gpus = self.args.gpus

        format = '%(asctime)s - %(name)s - %(message)s'
        logging.basicConfig(format=format, filename=os.path.join(self.args.save_path, "knn_result_log.txt"), level=logging.INFO)
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)

    def forward(self, input_ids, pinyin_ids):
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, pinyin_ids, attention_mask=attention_mask)

    def test_dataloader(self, ) -> DataLoader:
        dataset = NERDataset(directory=self.args.data_dir, prefix="test",
                                      vocab_file=os.path.join(self.args.bert_path, "vocab.txt"),
                                      max_length=self.args.max_length,
                                      config_path=os.path.join(self.args.bert_path, "config"),
                                      file_name=self.args.file_name)

        batch_size = self.args.batch_size
        data_sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                                drop_last=False)

        info = json.load(open(os.path.join(self.args.datastore_path, "datastore_info.json")))
        key_file = os.path.join(self.args.datastore_path, "keys.npy")
        keys = np.memmap(key_file, 
                     dtype=np.float32,
                     mode="r",
                     shape=(info['token_sum'], info['hidden_size']))
        keys_in_memory = np.zeros((info['token_sum'], info['hidden_size']), dtype=np.float32)
        keys_in_memory[:] = keys[:]
        
        self.keys = torch.from_numpy(keys_in_memory)
        
        val_file = os.path.join(self.args.datastore_path, "vals.npy")
        vals = np.memmap(val_file, 
                     dtype=np.int32,
                     mode="r",
                     shape=(info['token_sum'],))
        vals_in_memory = np.zeros((info['token_sum'],), dtype=np.int64)
        vals_in_memory[:] = vals[:]
        
        self.vals = torch.from_numpy(vals_in_memory)
        
        self.link_temperature = torch.tensor(self.args.link_temperature)

        self.link_ratio = torch.tensor(self.args.link_ratio)

        if (self.num_gpus):
            self.keys = self.keys.cuda()
            self.vals = self.vals.cuda()
            self.link_temperature = self.link_temperature.cuda()
            self.link_ratio = self.link_ratio.cuda()

        return dataloader

    def test_step(self, batch, batch_idx):
        input_ids, pinyin_ids, gold_labels = batch
        sequence_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, seq_len, 8)

        model_output = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids,)

        argmax_labels = self.postprocess_logits_to_labels(model_output.logits, model_output.hidden_states[-1])
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=sequence_mask)
        return {"confusion_matrix": confusion_matrix}

    def test_epoch_end(self, outputs):
        confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        if self.args.save_ner_prediction:
            precision, recall, f1, entity_tuple = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative, prefix="test")
            gold_entity_lst, pred_entity_lst = entity_tuple
            self.save_predictions_to_file(gold_entity_lst, pred_entity_lst)
        else:
            precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        tensorboard_logs = {"test_f1": f1}
        self.result_logger.info(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} , link_temperature: {self.link_temperature}, link_ratio: {self.link_ratio}")
        return {"test_log": tensorboard_logs, "test_f1": f1, "test_precision": precision, "test_recall": recall}

    def postprocess_logits_to_labels(self, logits, hidden):
        """input logits should in the shape [batch_size, seq_len, num_labels]"""
        probabilities = F.softmax(logits, dim=2) # shape of [batch_size, seq_len, num_labels]

        batch_size = hidden.shape[0]
        sent_len = hidden.shape[1]
        hidden_size = hidden.shape[-1]
        token_num = self.keys.shape[0]

        # cosine similarity
        knn_feats = self.keys.transpose(0, 1) # [feature_size, token_num]
        hidden = hidden.view(-1, hidden_size) # [bsz*sent_len, feature_size]
        sim = torch.mm(hidden, knn_feats) # [bsz*sent_len, token_num]
        norm_1 = (knn_feats ** 2).sum(dim=0, keepdim=True).sqrt() # [1, token_num]
        norm_2 = (hidden ** 2).sum(dim=1, keepdim=True).sqrt() # [bsz*sent_len, 1]
        scores = (sim / (norm_1 + 1e-10) / (norm_2 + 1e-10)).view(batch_size, sent_len, -1) # [bsz, sent_len, token_num]
        knn_labels = self.vals.view(1, 1, token_num).expand(batch_size, sent_len, token_num) # [bsz, sent_len, token_num]
        
        if (self.args.topk != -1 and scores.shape[-1] > self.args.topk):
            topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=self.args.topk)  # [bsz, sent_len, topk]
            scores = topk_scores
            knn_labels = knn_labels.gather(dim=-1, index=topk_idxs)  # [bsz, sent_len, topk]
        
        sim_probs = torch.softmax(scores / self.link_temperature, dim=-1) # [bsz, sent_len, token_num]
        
        knn_probabilities = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1, 1, self.num_labels])  # [bsz, sent_len, num_labels]
        knn_probabilities = knn_probabilities.scatter_add(dim=2, index=knn_labels, src=sim_probs) # [bsz, sent_len, num_labels]

        probabilities = self.link_ratio*knn_probabilities + (1-self.link_ratio)*probabilities

        argmax_labels = torch.argmax(probabilities, 2, keepdim=False) # [bsz, sent_len]
        return argmax_labels

    def save_save_predictions_to_file(self, gold_entity_lst, pred_entity_lst, prefix="test"):
        dataset = self._load_dataset(prefix=prefix)
        data_items = dataset.data_items

        save_file_path = os.path.join(self.args.save_path, "test_predictions.txt")
        print(f"INFO -> write predictions to {save_file_path}")
        with open(save_file_path, "w") as f:
            for gold_label_item, pred_label_item, data_item in zip(gold_entity_lst, pred_entity_lst, data_items):
                data_tokens = data_item[0]
                f.write("=!" * 20+"\n")
                f.write("".join(data_tokens)+"\n")
                f.write(gold_label_item+"\n")
                f.write(pred_label_item+"\n")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, )
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--file_name", default="", type=str, help="use for truncated sets.")
    parser.add_argument("--save_ner_prediction", action="store_true", help="only work for test.")
    parser.add_argument("--path_to_model_hparams_file", default="", type=str, help="use for evaluation")
    parser.add_argument("--checkpoint_path", default="", type=str, help="use for evaluation.")
    parser.add_argument("--datastore_path", default="", type=str, help="use for saving datastore.")
    parser.add_argument("--link_temperature", default=1.0, type=float, help="temperature used by edge linking.")
    parser.add_argument("--link_ratio", default=0.0, type=float, help="ratio of vocab probs predicted by edge linking.")
    parser.add_argument("--topk", default=64, type=int, help="use topk-scored neighbor tgt nodes for link prediction and probability compuation.")

    return parser

def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ner_model = NERTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                            hparams_file=args.path_to_model_hparams_file,
                                                            map_location=None,
                                                            batch_size=args.batch_size)

    model = KNNNERTask(args)
    model.model = ner_model.model

    trainer = Trainer.from_argparse_args(args, deterministic=True)

    trainer.test(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()



