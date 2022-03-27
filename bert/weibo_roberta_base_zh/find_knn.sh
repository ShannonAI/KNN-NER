export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/weibo"
FILE_NAME="all.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_result/weibo_roberta_large"
BERT_PATH="/userhome/shuhe/ner/models/chinese-roberta-wwm-ext"
PARAMS_FILE="/userhome/shuhe/ner/chinese_result/weibo_roberta_large/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/chinese_result/weibo_roberta_large/checkpoint/epoch=9_v1.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/weibo/train-datastore-roberta-large"

CUDA_VISIBLE_DEVICES=1 python ./build_datastore.py \
--bert_path $BERT_PATH \
--batch_size 5 \
--workers 16 \
--max_length 512 \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--path_to_model_hparams_file $PARAMS_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--datastore_path $DATASTORE_PATH \
--gpus="1"