export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/weibo"
FILE_NAME="all.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_bert_result/weibo_base"
BERT_PATH="/userhome/shuhe/ner/models/ChineseBERT-base"
PARAMS_FILE="/userhome/shuhe/ner/chinese_bert_result/weibo_base/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/chinese_bert_result/weibo_base/checkpoint/epoch=7_v0.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/weibo/train-datastore-chineseBERT-base"


CUDA_VISIBLE_DEVICES=1 python ./tasks/KNN-NER/build_datastore.py \
--bert_path $BERT_PATH \
--batch_size 10 \
--workers 4 \
--max_length 512 \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--path_to_model_hparams_file $PARAMS_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--datastore_path $DATASTORE_PATH \
--gpus="1"