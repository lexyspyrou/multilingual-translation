#!/usr/bin/env bash

arch=${arch:-transformer_iwslt_de_en_small}
exp_name=${exp_name:-train_distill_student}
extra_args=${extra_args:-}
sources=${sources:-eng}
targets=${targets:-hin}
max_tokens=${max_tokens:-256}


export PYTHONPATH=.

python train.py data-bin/ted_8_diverse/  \
  --task universal_translation \
  -a $arch \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.99)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0005 --stop-min-lr 1e-09 \
  --log-format json \
  --save-interval-updates 2000 \
  --log-interval 50 \
  --dropout 0.3 --weight-decay 0.0 --criterion distill_label_smoothed_cross_entropy --label-smoothing 0.1 \
  --fix-batches-to-gpus \
  --patience 100 \
  --max-source-positions 150 --max-target-positions 150 \
  --sources=$sources --targets=$targets \
  --max-update 300000 \
  --max-tokens=$max_tokens \
  --save-dir checkpoints/$exp_name $extra_args
