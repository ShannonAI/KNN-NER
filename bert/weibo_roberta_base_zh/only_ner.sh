export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/weibo"
FILE_NAME="all.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_result/weibo_roberta_large"
BERT_PATH="/userhome/shuhe/ner/models/chinese-roberta-wwm-ext"

CUDA_VISIBLE_DEVICES=1 python ./ner_trainer.py \
--lr 2e-5 \
--max_epochs 15 \
--max_length 512 \
--weight_decay 0.001 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.1  \
--train_batch_size 8 \
--accumulate_grad_batches 2 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--optimizer torch.adam \
--gpus="1"
