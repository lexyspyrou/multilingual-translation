#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB
# 	  --max-epoch 40 \
# 	  --distributed-world-size 1 \ - I think this refers to the number of GPUs
# changed epochs from 40 to 25

MODEL_DIR=checkpoints/related_ted8_m2o/latent_depth/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log


lang_pairs_str="bel-eng,rus-eng"

fairseq-train data-bin/ted_8_related/ \
  --user-dir examples/latent_depth/latent_depth_src \
  --lang-pairs "${lang_pairs_str}" \
  --arch latent_multilingual_transformer \
  --task multilingual_translation_latent_depth \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --share-encoders \
  --share-decoders \
  --decoder-langtok \
  --share-decoder-input-output-embed \
  --dropout 0.3 --attention-dropout 0.3 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --stop-min-lr 1e-9 --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --max-tokens 1000 --update-freq 1  \
  --lr 0.0015 \
  --clip-norm 1.0 \
  --seed 2 \
  --ddp-backend=legacy_ddp \
  --encoder-layers 4 \
  --decoder-layers 8 \
  --decoder-latent-layer \
  --sparsity-weight 0.1 \
  --anneal-updates 5000 \
  --soft-update 500  \
  --target-layers 4 \
  --share-weight 0.1 \
  --max-epoch 25
