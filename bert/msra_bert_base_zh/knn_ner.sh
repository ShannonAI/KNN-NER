export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/msra"
FILE_NAME="ner.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/msra_bert_base"
BERT_PATH="/userhome/shuhe/ner/models/bert-base-chinese"
PARAMS_FILE="/userhome/shuhe/ner/chinese_result/xiaya_dict/msra_bert_base/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/msra_bert_base/checkpoint/epoch=3.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/msra/train-datastore-bert-base"
link_temperature=0.17
link_ratio=0.8
topk=1024

CUDA_VISIBLE_DEVICES=0 python ./knn_ner_trainer.py \
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