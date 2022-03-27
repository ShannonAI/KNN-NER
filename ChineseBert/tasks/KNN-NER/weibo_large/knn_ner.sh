export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/weibo"
FILE_NAME="all.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_bert_result/weibo_large"
BERT_PATH="/userhome/shuhe/ner/models/ChineseBERT-large"
PARAMS_FILE="/userhome/shuhe/ner/chinese_bert_result/weibo_large/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/chinese_bert_result/weibo_large/checkpoint/epoch=4_v1.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/weibo/train-datastore-chineseBERT-large"
link_temperature=0.21
link_ratio=0.31
topk=-1

CUDA_VISIBLE_DEVICES=1 python ./tasks/KNN-NER/knn_ner_trainer.py \
--bert_path $BERT_PATH \
--batch_size 1 \
--workers 4 \
--max_length 512 \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--path_to_model_hparams_file $PARAMS_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--datastore_path $DATASTORE_PATH \
--save_path $SAVE_PATH \
--link_temperature $link_temperature \
--link_ratio $link_ratio \
--topk $topk \
--gpus="1"