#!/usr/bin/env bash

#data_dir=${data_dir:-iwslt}
hparams=${hparams:-}
arch=${arch:-transformer_iwslt_de_en_small}
update_freq=${update_freq:-1}
max_tokens=${max_tokens:-256}
sources=${sources:-eng}
targets=${targets:-mar}
exp_name=${exp_name:-teacher_expert/$sources-$targets}

# set to universal translation maybe, or take the knowledge distillation from it!!
export PYTHONPATH=.

python train.py data-bin/ted_8_diverse/  \
  --task translation \
  --arch transformer_iwslt_de_en_small \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0005 \
  --log-format json \
  --distributed-world-size 1 \
  --log-interval 50 \
  --max-epoch 2 \
  --no-epoch-checkpoints \
  --save-teacher-output \
  --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --fix-batches-to-gpus \
  --source-lang $sources \
  --distill-topk 4 \
  --target-lang $targets \
  --max-tokens=$max_tokens --update-freq=$update_freq \
  --max-update 300000 \
  --save-dir checkpoints/$exp_name $hparams