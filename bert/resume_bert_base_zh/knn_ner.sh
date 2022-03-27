export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/resume"
FILE_NAME="word.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/resume_bert_base"
BERT_PATH="/userhome/shuhe/ner/models/bert-base-chinese"
PARAMS_FILE="/userhome/shuhe/ner/chinese_result/xiaya_dict/resume_bert_base/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/chinese_result/xiaya_dict/resume_bert_base/checkpoint/epoch=6_v2.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/resume/train-datastore-bert-base"
link_temperature=0.1
link_ratio=0.1
topk=-1

for link_ratio in 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85;do
for link_temperature in 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15;do
CUDA_VISIBLE_DEVICES=1 python ./knn_ner_trainer.py \
--bert_path $BERT_PATH \
--batch_size 1 \
--workers 16 \
--max_length 512 \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--path_to_model_hparams_file $PARAMS_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--datastore_path $DATASTORE_PATH \
--save_path $SAVE_PATH \
--link_temperature $link_temperature \
--link_ratio $link_ratio \
--topk $topk \
--gpus="1"
done
done
