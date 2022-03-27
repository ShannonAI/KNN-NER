export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/ontonote4"
FILE_NAME="char.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/ontonote_roberta_large"
BERT_PATH="/userhome/shuhe/ner/models/chinese-roberta-wwm-ext-large"
PARAMS_FILE="/userhome/shuhe/ner/chinese_result/xiaya_dict/ontonote_roberta_large/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/ontonote_roberta_large/checkpoint/epoch=4_v2.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/ontonote4/train-datastore-roberta-large"
link_temperature=0.063
link_ratio=0.14
topk=2048

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