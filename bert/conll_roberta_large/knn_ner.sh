export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/knn_ner/en_roberta_data/en_conll03"
FILE_NAME="word.bmes"
SAVE_PATH="/nfs1/shuhe/knn_ner/result/conll_roberta_large_drop0.2_5e_5"
BERT_PATH="/nfs1/shuhe/knn_ner/roberta_model/roberta-large"
PARAMS_FILE="/nfs1/shuhe/knn_ner/result/conll_roberta_large_drop0.2/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/nfs1/shuhe/knn_ner/result/conll_roberta_large_drop0.2/checkpoint/epoch=32_v1.ckpt"
DATASTORE_PATH="/nfs1/shuhe/knn_ner/en_roberta_data/en_conll03/train-datastore-large"
link_temperature=0.001
link_ratio=0.24
topk=-1

CUDA_VISIBLE_DEVICES=0 python ./knn_ner_trainer.py \
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
--en_roberta \
--gpus="1"

TEST F1: 0.9181528091430664, Precision: 0.9164210557937622, Recall: 0.9198911190032959
TEST F1: 0.8736943006515503, Precision: 0.8658629059791565, Recall: 0.8816685676574707