export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/msra"
FILE_NAME="ner.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_bert_result/msra_base"
BERT_PATH="/userhome/shuhe/ner/models/ChineseBERT-base"

CUDA_VISIBLE_DEVICES=0 python ./tasks/KNN-NER/ner_trainer.py \
--lr 2e-5 \
--max_epochs 40 \
--max_length 512 \
--weight_decay 0.01 \
--hidden_dropout_prob 0.2 \
--warmup_proportion 0.001  \
--train_batch_size 26 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--precision=16 \
--gpus="1"
