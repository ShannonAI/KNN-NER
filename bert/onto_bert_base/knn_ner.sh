export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/en_data/ontonotes5"
FILE_NAME="word.bmes"
SAVE_PATH="/userhome/shuhe/ner/en_result/ontonotes_bert_base_drop_0.2"
BERT_PATH="/userhome/shuhe/ner/models/bert-base-cased"
PARAMS_FILE="/userhome/shuhe/ner/en_result/ontonotes_bert_base_drop_0.2/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/en_result/ontonotes_bert_base_drop_0.2/checkpoint/epoch=6_v0.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/en_data/ontonotes5/train-datastore-base"
link_temperature=0.04
link_ratio=0.43
topk=-1

CUDA_VISIBLE_DEVICES=1 python ./knn_ner_trainer.py \
--bert_path $BERT_PATH \
--batch_size 1 \
--workers 16 \
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