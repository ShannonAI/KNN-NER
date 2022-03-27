export PYTHONPATH="$PWD"

DARA_DIR="/data/wangshuhe/ner/en_ner_data/ontonotes5"
FILE_NAME="word.bmes"
SAVE_PATH="/data/wangshuhe/ner/bert_result/en_ontonotes"
BERT_PATH="/data/wangshuhe/ner/models/bert-base-cased"

echo "getting ner labels ..."
python ./get_labels.py --data-dir $DARA_DIR --file-name $FILE_NAME

CUDA_VISIBLE_DEVICES=0 python ./ner_trainer.py \
--lr 3e-5 \
--max_epochs 5 \
--max_length 512 \
--weight_decay 0.001 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.1  \
--train_batch_size 16 \
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