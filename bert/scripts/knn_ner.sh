export PYTHONPATH="$PWD"

DARA_DIR="/data/wangshuhe/ner/zh_ner_data/ontonote4"
FILE_NAME="char.bmes"
SAVE_PATH="/data/wangshuhe/ner/bert_result/onto_notes"
BERT_PATH="/data/wangshuhe/ner/models/bert-base-chinese"
PARAMS_FILE="/data/wangshuhe/ner/bert_result/onto_notes/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/data/wangshuhe/ner/bert_result/onto_notes/checkpoint/epoch=2_v1.ckpt"
DATASTORE_PATH="/data/wangshuhe/ner/zh_ner_data/ontonote4/train-datastore"
link_temperature=0.1
link_ratio=0.1
topk=64

for link_ratio in 0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91;do
for link_temperature in 0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91;do
CUDA_VISIBLE_DEVICES=0 python ./knn_ner_trainer.py \
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
done
done