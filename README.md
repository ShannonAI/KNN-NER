# kNN-NER: Named Entity Recognition with Nearest Neighbor Search

The repository contains the code of the recent research [KNN-NER](https://arxiv.org/abs/2203.17103).

**kNN-NER: Named Entity Recognition with Nearest Neighbor Search**

Shuhe Wang, Xiaoya Li, Yuxian Meng, Rongbin Ouyang, Jiwei Li, Guoyin Wang

If you find this repo helpful, please cite the following:
```
@article{wang2022k,
  title={$ k $ NN-NER: Named Entity Recognition with Nearest Neighbor Search},
  author={Wang, Shuhe and Li, Xiaoya and Meng, Yuxian and Zhang, Tianwei and Ouyang, Rongbin and Li, Jiwei and Wang, Guoyin},
  journal={arXiv preprint arXiv:2203.17103},
  year={2022}
}
```

## Requirements
1. python 3.6+
2. If you are working on a GPU machine, please install GPU version torch>=1.7.1, more details for PyTorch can be found on the [Official Website](https://pytorch.org/get-started/previous-versions/)
   
We provide two kinds of vanilla NER model for kNN-NER:
1. BERT and RoBERTa: Requirements for BERT and RoBERTa are same as the repo [MRC](https://github.com/ShannonAI/mrc-for-flat-nested-ner), you can run the command `pip install -r ./bert/requirements.txt` or look at [MRC](https://github.com/ShannonAI/mrc-for-flat-nested-ner) for more details.
2. ChineseBERT: Requirements for ChineseBERT are same as the repo [ChineseBERT](https://github.com/ShannonAI/ChineseBert), you can run the command `pip install -r ./ChineseBert/requirements.txt` or look at [ChineseBERT](https://github.com/ShannonAI/ChineseBert) for more details.

## Datasets
The preprocessed datasets used for KNN-NER can be found [here](https://drive.google.com/drive/folders/1HbxoGLY3n0BernLqWmEEH3NpMvtFNCY2?usp=sharing). Each dataset is splited into three fileds `train/valid/test`. The file `ner_labels.txt` in each dataset contains all the labels within it and you can generate it by running the script `python ./get_labels.py --data-dir DATADIR --file-name NAME`.

## The Vanilla NER model
### Pre-trained NER Model
You can direct download a pre-trained ner model as the vanilla NER model and only follow the two step **Building Datastore** and **Inference** to reproduce the results of KNN-NER.
#### Based on BERT
For pre-trained BERT NER model, you can use [**Baseline:BERT-Tagger**](https://github.com/ShannonAI/mrc-for-flat-nested-ner).
#### Based on ChineseBERT 
For pre-trained ChineseBERT NER model, you can use [**NER Task**](https://github.com/ShannonAI/ChineseBert).

### Training Models For NER
#### Downloading Backbone Models
There are five version models for Chinese datasets and three version models for English datasets, you can download them following:
##### Zh Models
* BERT-Base: https://huggingface.co/bert-base-chinese
* RoBERTa-Base: https://huggingface.co/hfl/chinese-roberta-wwm-ext
* RoBERTa-Large: https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
* ChineseBERT-Base: https://huggingface.co/ShannonAI/ChineseBERT-base
* ChineseBERT-Large: https://huggingface.co/ShannonAI/ChineseBERT-large

##### En Models
* BERT-Base: https://huggingface.co/bert-base-cased
* BERT-Large: https://huggingface.co/bert-large-cased
* RoBERTa-Large: https://huggingface.co/roberta-large

#### Training
You can train a new NER model following the script `./ChineseBert/KNN-NER/DATASET_NAME/only_ner.sh` with ChineseBERT as the backbone and the script `./bert/DATASET_NAME/only_ner.sh` with BERT or RoBERTa as the backbone. The *DATASET_NAME* is the combination of the used dataset and backbone model, such as *weibo_bert_base_zh* means training a NER model for dataset **Weibo** with model **bert-base-chinese** as the backbone. Note that you need to change `DARA_DIR`, `FILE_NAME`, `SAVE_PATH` and `BERT_PATH` to your own path.

## Building Datastore
To build datastore for you trained or pre-trained NER model, you can run the script `./ChineseBert/KNN-NER/DATASET_NAME/find_knn.sh` or `./bert/DATASET_NAME/find_knn.sh`. The meaning of *DATASET_NAME* is same as the above **Training** setp, wihch is the combination of the used dataset and backbone model.

## Inference
Code for inference using the KNN-NER model can be found in `./ChineseBert/KNN-NER/DATASET_NAME/knn_ner.sh` or `./bert/DATASET_NAME/knn_ner.sh`.
### Results
**Date 2022.03.29, the results are same as the paper [here]().**
#### Chinese OntoNotes 4.0
| Model | Test Precision | Test Recall | Test F1 |
| - | - | - | - |
| ***Base Model*** | ---- | ---- | ---- |
| BERT-Base | 78.01 | 80.35 | 79.16 |
| BERT-Base+kNN | 80.23 | 81.60 | **80.91 (+1.75)** |
| RoBERTa-Base | 80.43 | 80.30 | 80.37 |
| RoBERTa-Base+kNN | 79.65 | 82.60 | **81.10 (+0.73)** |
| ChineseBERT-Base | 80.03 | 83.33 | 81.65 |
| ChineseBERT-Base+kNN | 81.43 | 82.58 | **82.00 (+0.35)** |
| ***Large Model*** | ---- | ---- | ---- |
| RoBERTa-Large | 80.72 | 82.07 | 81.39 |
| RoBERTa-Large+kNN | 79.87 | 83.17 | **81.49 (+0.10)** |
| ChineseBERT-Large | 80.77 | 83.65 | 82.18 |
| ChineseBERT-Large+kNN | 81.68 | 83.46 | **82.56 (+0.38)** |
#### Chinese MSRA
| Model | Test Precision | Test Recall | Test F1 |
| - | - | - | - |
| ***Base Model*** | ---- | ---- | ---- |
| BERT-Base | 94.97 | 94.62 | 94.80 |
| BERT-Base+kNN | 95.34 | 94.64 | **94.99 (+0.19)** |
| RoBERTa-Base | 95.27 | 94.66 | 94.97 |
| RoBERTa-Base+kNN | 95.47 | 94.79 | **95.13 (+0.16)** |
| ChineseBERT-Base | 95.39 | 95.39 | 95.39 |
| ChineseBERT-Base+kNN | 95.73 | 95.27 | **95.50 (+0.11)** |
| ***Large Model*** | ---- | ---- | ---- |
| RoBERTa-Large | 95.87 | 94.89 | 95.38 |
| RoBERTa-Large+kNN | 95.96 | 95.02 | **95.49 (+0.11)** |
| ChineseBERT-Large | 95.61 | 95.61 | 95.61 |
| ChineseBERT-Large+kNN | 95.83 | 95.68 | **95.76 (+0.15)** |
#### Chinese Weibo
| Model | Test Precision | Test Recall | Test F1 |
| - | - | - | - |
| ***Base Model*** | ---- | ---- | ---- |
| BERT-Base | 67.12 | 66.88 | 67.33 |
| BERT-Base+kNN | 70.07 | 67.87 | **68.96 (+1.63)** |
| RoBERTa-Base | 68.49 | 67.81 | 68.15 |
| RoBERTa-Base+kNN | 67.52 | 69.81 | **68.65 (+0.50)** |
| ChineseBERT-Base | 68.27 | 69.78 | 69.02 |
| ChineseBERT-Base+kNN | 68.97 | 73.71 | **71.26 (+2.24)** |
| ***Large Model*** | ---- | ---- | ---- |
| RoBERTa-Large | 66.74 | 70.02 | 68.35 |
| RoBERTa-Large+kNN | 69.36 | 70.53 | **69.94 (+1.59)** |
| ChineseBERT-Large | 68.75 | 72.97 | 70.80 |
| ChineseBERT-Large+kNN | 75.00 | 69.29 | **72.03 (+1.23)** |
#### English CoNLL 2003
| Model | Test Precision | Test Recall | Test F1 |
| - | - | - | - |
| ***Base Model*** | ---- | ---- | ---- |
| BERT-Base | 90.69 | 91.96 | 91.32 |
| BERT-Base+kNN | 91.50 | 91.58 | **91.54 (+0.22)** |
| ***Large Model*** | ---- | ---- | ---- |
| BERT-Large | 91.54 | 92.79 | 92.16 |
| BERT-Large+kNN | 92.26 | 92.43 | **92.40 (+0.24)** |
| RoBERTa-Large | 92.77 | 92.81 | 92.76 |
| RoBERTa-Large+kNN | 92.82 | 92.99 | **92.93 (+0.17)** |
#### English OntoNotes 5.0
| Model | Test Precision | Test Recall | Test F1 |
| - | - | - | - |
| ***Base Model*** | ---- | ---- | ---- |
| BERT-Base | 85.09 | 85.99 | 85.54 |
| BERT-Base+kNN | 85.27 | 86.13 | **85.70 (+0.16)** |
| ***Large Model*** | ---- | ---- | ---- |
| BERT-Large | 85.84 | 87.61 | 86.72 |
| BERT-Large+kNN | 85.92 | 87.84 | **86.87 (+0.15)** |
| RoBERTa-Large | 86.59 | 88.17 | 87.37 |
| RoBERTa-Large+kNN | 86.73 | 88.29 | **87.51 (+0.14)** |

## Contact
If you have any question about our paper/code/modal/data...

Please feel free to discuss through github issues or emails.

You can send emails to [shuhe_wang@shannonai.com](shuhe_wang@shannonai.com)