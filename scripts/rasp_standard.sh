#!/bin/bash 

DATASET="sort"
D_MODEL=64
N_LAYERS=3
N_HEADS=8
SEED=0
DO_GLOVE=0

python3 -m src.run \
     --dataset "${DATASET}" \
     --standard \
     --dataset_size 20000 \
     --vocab_size 8 \
     --min_length 1 \
     --max_length 8 \
     --n_epochs 30 \
     --batch_size 512 \
     --lr "5e-2" \
     --d_model "${D_MODEL}" \
     --n_heads "${N_HEADS}" \
     --n_layers "${N_LAYERS}" \
     --d_mlp 64 \
     --dropout 0.0 \
     --max_grad_norm 5.0 \
     --do_lower 0 \
     --replace_numbers 0 \
     --do_glove "${DO_GLOVE}" \
     --pool_outputs 1 \
     --seed "${SEED}" \
     --save \
     --gpu_uuid "GPU-96f047f6-0272-02db-e089-678debe69d9b" \
     --output_dir "output/rasp/${DATASET}/standard_transformer/dmodel${D_MODEL}nheads${N_CAT}nlayers${N_LAYERS}/s${SEED}";