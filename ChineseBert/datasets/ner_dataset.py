#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file  : ner_dataset.py
@author: shuhe wang
@contact : shuhe_wang@shannonai.com
@date  : 2021/07/07 16:11
@version: 1.0
@desc  :
"""

import os
import json
from typing import List
from pypinyin import pinyin, Style

import torch
import tokenizers
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer


class NERDataset(Dataset):

    __ner_labels = None

    """the Dataset Class for NER task."""
    def __init__(self, directory, prefix, vocab_file, config_path, max_length=512, file_name="char.bmes"):
        """
        Args:
            directory: str, path to data directory.
            prefix: str, one of [train/dev/test]
            vocab_file: str, path to the vocab file for model pre-training.
            config_path: str, config_path must contain [pinyin_map.json, id2pinyin.json, pinyin2tensor.json]
            max_length: int,
        """
        super().__init__()
        self.max_length = max_length
        data_file_path = os.path.join(directory, "{}.{}".format(prefix, file_name))
        self.data_items = NERDataset._read_conll(data_file_path)
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.label_to_idx = {label_item: label_idx for label_idx, label_item in enumerate(NERDataset.get_labels(os.path.join(directory, "ner_labels.txt")))}

        # load pinyin map dict
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        token_sequence, label_sequence = data_item[0], data_item[1]
        label_sequence = [self.label_to_idx[label_item] for label_item in label_sequence]
        token_sequence = "".join(token_sequence[: self.max_length - 2])
        #token_sequence = token_sequence[: self.max_length - 2]
        label_sequence = label_sequence[: self.max_length - 2]
        
        # convert string to ids
        tokenizer_output = self.tokenizer.encode(token_sequence)
        # example of tokenizer_output ->
        # Encoding(num_tokens=77, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
        bert_tokens = tokenizer_output.ids
        label_sequence = self._update_labels_using_tokenize_offsets(tokenizer_output.offsets, label_sequence)
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(token_sequence, tokenizer_output)
        '''
        bert_tokens = [self.tokenizer.token_to_id(token_str) if self.tokenizer.token_to_id(token_str) is not None else self.tokenizer.token_to_id('[UNK]') for token_str in token_sequence]
        bert_tokens = [self.tokenizer.token_to_id('[CLS]')] + bert_tokens + [self.tokenizer.token_to_id('[SEP]')]
        label_sequence = [self.label_to_idx["O"]] + label_sequence + [self.label_to_idx["O"]]
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(token_sequence)
        '''
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        assert len(bert_tokens) == len(label_sequence)
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor(label_sequence)
        return input_ids, pinyin_ids, label

    def _update_labels_using_tokenize_offsets(self, offsets, original_sequence_labels):
        """part of offset sequence [(51, 52), (52, 54)] -> (token index after tokenized, original token index)"""
        update_sequence_labels = []
        for offset_idx, offset_item in enumerate(offsets):
            if offset_idx == 0 or offset_idx == (len(offsets) - 1):
                continue
            update_index, origin_index = offset_item
            current_label = original_sequence_labels[origin_index-1]
            update_sequence_labels.append(current_label)
        update_sequence_labels = [self.label_to_idx["O"]] + update_sequence_labels + [self.label_to_idx["O"]]
        return update_sequence_labels

    @classmethod
    def get_labels(cls, file_path):
        """gets the list of labels for this data set."""
        
        if (cls.__ner_labels is None):
            cls.__ner_labels = []
            file = open(file_path, "r")
            for line in file:
                if (len(line.strip()) != 0):
                    cls.__ner_labels.append(line.strip())

        return cls.__ner_labels
        
        #return ["O", "S-LOC", "B-LOC", "M-LOC", "E-LOC", "S-PER", "B-PER", "M-PER", "E-PER", "S-GPE", "B-GPE", "M-GPE", "E-GPE", "S-ORG", "B-ORG", "M-ORG", "E-ORG"]

    @staticmethod
    def _read_conll(input_file, delimiter=" "):
        """load ner dataset from CoNLL-format files."""
        dataset_item_lst = []
        with open(input_file, "r", encoding="utf-8") as r_f:
            datalines = r_f.readlines()

        cached_token, cached_label = [], []
        for idx, data_line in enumerate(datalines):
            data_line = data_line.strip()
            if len(data_line) == 0:
                if (len(cached_token) != 0 and len(cached_label) != 0):
                    dataset_item_lst.append([cached_token, cached_label])
                cached_token, cached_label = [], []
            else:
                token_label = data_line.split(delimiter)
                token_data_line, label_data_line = token_label[0], token_label[1]
                cached_token.append(token_data_line)
                cached_label.append(label_data_line)
        return dataset_item_lst
    
    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # get pinyin in a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids.
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids
    '''
    def convert_sentence_to_pinyin_ids(self, sentence: str) -> List[List[int]]:
        # get pinyin in a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids.
        pinyin_ids = []
        for idx_ in range(len(sentence)):
            if (idx_ in pinyin_locs):
                pinyin_ids.append(pinyin_locs[idx_])
            else:
                pinyin_ids.append([0] * 8)
        pinyin_ids = [[0] * 8] + pinyin_ids + [[0] * 8]

        return pinyin_ids
    '''