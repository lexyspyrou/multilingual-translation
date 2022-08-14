#!/usr/bin/env bash

MODEL_DIR=checkpoints/related_ted8_o2m/latent_depth/original_rerun
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"
echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

python train.py data-bin/ted_8_related/  \
  --user-dir examples/latent_depth/latent_depth_src \
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
  --update-freq 1  \
  --lr 0.0015 \
  --clip-norm 1.0 \
  --seed 2 \
  --max-epoch 40 \
  --max-update 64500 \
  --ddp-backend=legacy_ddp \
  --decoder-latent-layer \
  --sparsity-weight 0.1 \
  --anneal-updates 5000 \
  --soft-update 500  \
  --distributed-world-size 4 \
  --save-teacher-output \
  --fix-batches-to-gpus \
  --distill-topk 4 \
  --save-dir $MODEL_DIR \
  --tensorboard-logdir $MODEL_DIR \
  --log-interval 100 >> $MODEL_DIR/train.log 2>&1 \
  --skip-invalid-size-inputs-valid-test \
	--encoder-layers 6 \
  --decoder-layers 12 \
  --max-tokens 2048 \
  --target-layers 6 \
  --lang-pairs  "eng-aze,eng-bel,eng-ces,eng-glg,eng-por,eng-rus,eng-slk,eng-tur" \
  --restore-file $MODEL_DIR/checkpoint_best.pt


