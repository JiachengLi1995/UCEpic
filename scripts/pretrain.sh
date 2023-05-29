CUDA_VISIBLE_DEVICES=3,5 torchrun --nproc_per_node 2 pretrain.py \
--pregenerated_data ./processed \
--bert_model roberta-base \
--output_dir ./wiki \
--epochs 2  \
--fp16 \
--train_batch_size 512 \
--output_step 100000 \
--learning_rate 5e-5 \


# CUDA_VISIBLE_DEVICES=3 python pretrain.py \
# --pregenerated_data ./processed \
# --bert_model roberta-base \
# --output_dir ./wiki \
# --epochs 2  \
# --fp16 \
# --train_batch_size 512 \
# --output_step 100000 \
# --learning_rate 5e-5 