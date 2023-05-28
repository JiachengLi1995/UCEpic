DATA=beer
CKPT=checkpoints/dev/update_7fbb99e8/output/beer/seed_42_2022-08-03-15-44-45/checkpoint-1048
EXT=1
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
 
CUDA_VISIBLE_DEVICES=0 python tools/compute_scores.py ${CKPT}/generated_${EXT}_bert_${CONTROL}_${NUM}.json
