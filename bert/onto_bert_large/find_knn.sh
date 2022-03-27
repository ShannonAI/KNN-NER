export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/en_data/ontonotes5"
FILE_NAME="word.bmes"
SAVE_PATH="/userhome/shuhe/ner/en_result/ontonotes_bert_large_drop_0.2_weight_decay_0.01_warmup_proportion_0.2"
BERT_PATH="/userhome/shuhe/ner/models/bert-large-cased"
PARAMS_FILE="/userhome/shuhe/ner/en_result/ontonotes_bert_large_drop_0.2_weight_decay_0.01_warmup_proportion_0.2/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/en_result/ontonotes_bert_large_drop_0.2_weight_decay_0.01_warmup_proportion_0.2/checkpoint/epoch=9.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/en_data/ontonotes5/train-datastore-large"

CUDA_VISIBLE_DEVICES=0 python ./build_datastore.py \
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