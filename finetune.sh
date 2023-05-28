DATA=${1}

CUDA_VISIBLE_DEVICES=7 python finetune.py \
    --path data/${DATA} \
    --output_dir output/${DATA} \
    --epochs 10 \
    --gradient_accumulation_steps 4 \
    --train_batch_size 128 \
    --pretrained_model pretrained_model/wiki/checkpoint-15174 \
    --fp16 \