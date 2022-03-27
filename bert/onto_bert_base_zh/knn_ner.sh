export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/ontonote4"
FILE_NAME="char.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/ontonote_bert_base"
BERT_PATH="/userhome/shuhe/ner/models/bert-base-chinese"
PARAMS_FILE="/userhome/shuhe/ner/chinese_result/xiaya_dict/ontonote_bert_base/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/ontonote_bert_base/checkpoint/epoch=6_v1.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/ontonote4/train-datastore-bert-base"
link_temperature=0.031
link_ratio=0.81
topk=256

CUDA_VISIBLE_DEVICES=1 python ./knn_ner_trainer.py \
--bert_path $BERT_PATH \
--batch_size 2 \
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
--precision=16 \
--gpus="1"