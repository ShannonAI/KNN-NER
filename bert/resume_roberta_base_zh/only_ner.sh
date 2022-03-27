export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/resume"
FILE_NAME="word.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/resume_roberta_base"
BERT_PATH="/userhome/shuhe/ner/models/chinese-roberta-wwm-ext"


CUDA_VISIBLE_DEVICES=1 python ./ner_trainer.py \
--lr 3e-5 \
--max_epochs 10 \
--max_length 512 \
--weight_decay 0.01 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.001  \
--train_batch_size 16 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--precision=16 \
--language zh \
--gpus="1"
