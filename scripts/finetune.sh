DATA=${1}

CUDA_VISIBLE_DEVICES=3 python finetune.py \
    --path data/${DATA} \
    --output_dir output/${DATA} \
    --epochs 10 \
    --gradient_accumulation_steps 4 \
    --train_batch_size 128 \
    --pretrained_model ./wiki_ckpt \
    --fp16 \