export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/knn_ner/en_roberta_data/en_conll03"
FILE_NAME="word.bmes"
SAVE_PATH="/nfs1/shuhe/knn_ner/result/conll_roberta_large_drop0.2_5e_5"
BERT_PATH="/nfs1/shuhe/knn_ner/roberta_model/roberta-large"
PARAMS_FILE="/nfs1/shuhe/knn_ner/result/conll_roberta_large_drop0.2/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/nfs1/shuhe/knn_ner/result/conll_roberta_large_drop0.2/checkpoint/epoch=32_v1.ckpt"
DATASTORE_PATH="/nfs1/shuhe/knn_ner/en_roberta_data/en_conll03/train-datastore-large"

CUDA_VISIBLE_DEVICES=3 python ./build_datastore.py \
--bert_path $BERT_PATH \
--batch_size 5 \
--workers 16 \
--max_length 512 \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--path_to_model_hparams_file $PARAMS_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--datastore_path $DATASTORE_PATH \
--en_roberta \
--gpus="1"