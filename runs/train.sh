#!/usr/bin/env bash

#data_dir=${data_dir:-iwslt}
hparams=${hparams:-}
update_freq=${update_freq:-1}
sources=${sources:-eng}
targets=${targets:-bel}
exp_name=${exp_name:-teacher_expert/multilingual}

# set to universal translation maybe, or take the knowledge distillation from it!!
export PYTHONPATH=.
#  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces"\
#  --task multilingual_translation_latent_depth \
#  --lang-pairs "eng-aze,eng-bel"\
#  --target-layers 1 \

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
  --max-epoch 4 \
  --ddp-backend=legacy_ddp \
  --decoder-latent-layer \
  --sparsity-weight 0.1 \
  --anneal-updates 5000 \
  --soft-update 500  \
  --distributed-world-size 1\
  --log-format json \
  --log-interval 50 \
  --no-epoch-checkpoints \
  --save-teacher-output \
  --fix-batches-to-gpus \
  --distill-topk 4 \
  --max-update 1000 \
  --max-source-positions 150 --max-target-positions 150 \
  --save-dir checkpoints/$exp_name $hparams \
  --tensorboard-logdir checkpoints/$exp_name $hparams \
  --skip-invalid-size-inputs-valid-test \
	--encoder-layers 1 \
  --decoder-layers 1 \
  --max-tokens 1024 \
  --target-layers 1 \
  --lang-pairs "eng-aze,eng-bel,eng-glg"\
  --encoder-embed-dim 16 \
  --decoder-embed-dim 16 \
  --encoder-ffn-embed-dim 16 \
  --decoder-ffn-embed-dim 16


#  --target-layers 1 \
#  --decoder-layers 1 \
#	--encoder-layers 1 \
#  --source-lang $sources \
#  --target-lang $targets \
#    --task translation \
