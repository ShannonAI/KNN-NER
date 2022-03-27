export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/en_data/ontonotes5"
FILE_NAME="word.bmes"
SAVE_PATH="/userhome/shuhe/ner/en_result/ontonotes_bert_base_drop_0.2"
BERT_PATH="/userhome/shuhe/ner/models/bert-base-cased"


CUDA_VISIBLE_DEVICES=1 python ./ner_trainer.py \
--lr 2e-5 \
--max_epochs 10 \
--max_length 512 \
--weight_decay 0.001 \
--hidden_dropout_prob 0.2 \
--warmup_proportion 0.1  \
--train_batch_size 32 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--optimizer torch.adam \
--classifier multi \
--precision=16 \
--gpus="1"