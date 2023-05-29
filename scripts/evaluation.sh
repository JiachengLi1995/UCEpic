DATA=yelp/finetune
CKPT=/data1/jiacheng/UCEpic/checkpoints/main/init_c93e8ef2/output/yelp/finetune/seed_42_2023-05-28-17-15-45/checkpoint-102
EXT=phrases
NUM=-1
CONTROL=hard

CUDA_VISIBLE_DEVICES=0 python finetune_inference.py \
    --pregenerated_data data/${DATA} \
    --checkpoint ${CKPT} \
    --max_seq_len 64 \
    --keywords end2end \
    --phrase_key ${EXT} \
    --key_num ${NUM} \
    --eval true \
    --control ${CONTROL}
 
CUDA_VISIBLE_DEVICES=0 python compute_scores.py ${CKPT}/generated_${EXT}_bert_${CONTROL}_${NUM}.json
