export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/weibo"
FILE_NAME="all.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_bert_result/weibo_base"
BERT_PATH="/userhome/shuhe/ner/models/ChineseBERT-base"

CUDA_VISIBLE_DEVICES=1 python ./tasks/KNN-NER/ner_trainer.py \
--lr 4e-5 \
--max_epochs 10 \
--max_length 150 \
--weight_decay 0.01 \
--hidden_dropout_prob 0.2 \
--warmup_proportion 0.02  \
--train_batch_size 2 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--optimizer torch.adam \
--classifier multi \
--gpus="1"
