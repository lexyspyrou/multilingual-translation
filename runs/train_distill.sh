#!/usr/bin/env bash

arch=${arch:-transformer_iwslt_de_en_small}
exp_name=${exp_name:-train_distill_student}
extra_args=${extra_args:-}
max_tokens=${max_tokens:-256}

#
#sources=${sources:-eng}
#targets=${targets:-hin}
data=${data:-teacher_expert/$sources-$targets}

export PYTHONPATH=.

python train.py data-bin/ted_8_related/  \
  --lang-pairs "eng-aze,eng-bel"\
  --arch latent_multilingual_transformer \
  --task multilingual_translation_latent_depth \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.99)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --distributed-world-size 1 \
  --lr 0.0005 --stop-min-lr 1e-09 \
  --log-format json \
  --distill-topk 4 \
  --max-epoch 2 \
  --ignore-unused-valid-subsets \
  --save-interval-updates 2000 \
  --tensorboard-logdir $MODEL_DIR/tensorboard_dir.log \
  --log-interval 50 \
  --dropout 0.3 --weight-decay 0.0 --criterion distill_label_smoothed_cross_entropy --label-smoothing 0.1 \
  --fix-batches-to-gpus \
  --patience 2 \
  --max-source-positions 150 --max-target-positions 150 \
  --max-update 1000 \
  --max-tokens=$max_tokens \
  --save-dir checkpoints/$exp_name   \
  --tensorboard-logdir checkpoints/$exp_name
