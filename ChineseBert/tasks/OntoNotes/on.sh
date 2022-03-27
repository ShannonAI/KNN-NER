export PYTHONPATH="$PWD"

CUDA_VISIBLE_DEVICES=3 python3 ./tasks/OntoNotes/OntoNotes_trainer.py \
--lr 3e-5 \
--max_epochs 5 \
--max_length 275 \
--weight_decay 0.002 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.002  \
--train_batch_size 15 \
--accumulate_grad_batches 2 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path /home/wangshuhe/shuhework/chinese_bert/ChineseBert/on_result \
--bert_path /home/wangshuhe/shuhework/chinese_bert/ChineseBERT-base \
--data_dir /data/wangshuhe/ner/ner/ontonote4 \
--gpus="1"