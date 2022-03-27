export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/msra"
FILE_NAME="ner.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/msra_bert_base"
BERT_PATH="/userhome/shuhe/ner/models/bert-base-chinese"

CUDA_VISIBLE_DEVICES=0 python ./ner_trainer.py \
--lr 3e-5 \
--max_epochs 5 \
--max_length 512 \
--weight_decay 0.01 \
--hidden_dropout_prob 0.2 \
--warmup_proportion 0.01  \
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
--language zh \
--gpus="1"
