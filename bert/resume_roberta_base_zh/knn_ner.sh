export PYTHONPATH="$PWD"

DARA_DIR="/userhome/shuhe/ner/ner_data/resume"
FILE_NAME="word.bmes"
SAVE_PATH="/userhome/shuhe/ner/chinese_result/resume_roberta_large"
BERT_PATH="/userhome/shuhe/ner/models/chinese-roberta-wwm-ext"
PARAMS_FILE="/userhome/shuhe/ner/chinese_result/resume_roberta_large/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/shuhe/ner/chinese_result/resume_roberta_large/checkpoint/epoch=10_v1.ckpt"
DATASTORE_PATH="/userhome/shuhe/ner/ner_data/resume/train-datastore-roberta-large"
link_temperature=0.1
link_ratio=0.1
topk=-1

for link_ratio in 0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91;do
for link_temperature in 0.001 0.011 0.021 0.031 0.041 0.051 0.061 0.071 0.081 0.091;do
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
