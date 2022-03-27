export PYTHONPATH="$PWD"

DARA_DIR="/nfs1/shuhe/knn_ner/en_roberta_data/ontonotes5"
FILE_NAME="word.bmes"
SAVE_PATH="/nfs1/shuhe/knn_ner/result/ontonotes_roberta_large_drop0.1"
BERT_PATH="/nfs1/shuhe/knn_ner/roberta_model/roberta-large"

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=3 python ./ner_trainer.py \
--lr 3e-5 \
--max_epochs 20 \
--max_length 512 \
--weight_decay 0.01 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.001  \
--train_batch_size 16 \
--accumulate_grad_batches 2 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--optimizer torch.adam \
--precision=16 \
--classifier multi \
--en_roberta \
--gpus="1"
