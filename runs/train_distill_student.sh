#!/usr/bin/env bash

MODEL_DIR=checkpoints/train_distill_student_diverse_SPARSE_alpha0
mkdir -p $MODEL_DIR
#  --lang-pairs  "eng-aze,eng-bel,eng-ces,eng-glg,eng-por,eng-rus,eng-slk,eng-tur" \
export PYTHONPATH=.

python train.py data-bin/ted_8_diverse/  \
  --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
  --arch latent_multilingual_transformer \
  --task multilingual_translation_latent_depth \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.99)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --distributed-world-size 1 \
  --lr 0.0005 --stop-min-lr 1e-09 \
  --log-format json \
  --distill-topk 4 \
  --share-encoders \
  --share-decoders \
  --decoder-langtok \
  --max-epoch 40 \
  --ignore-unused-valid-subsets \
  --tensorboard-logdir $MODEL_DIR/tensorboard_dir.log \
  --dropout 0.3 \
  --no-epoch-checkpoints \
  --ddp-backend=legacy_ddp \
  --weight-decay 0.0 \
  --criterion distill_label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --fix-batches-to-gpus \
  --patience 4 \
  --save-dir $MODEL_DIR \
  --tensorboard-logdir $MODEL_DIR \
  --log-interval 100 >> $MODEL_DIR/train.log 2>&1 \
	--encoder-layers 6 \
  --decoder-layers 6 \
  --max-tokens 2048 \
  --target-layers 3 \
  --skip-invalid-size-inputs-valid-test \


#train_distill_student_diverse_alpha0.9==SPARSEEE
#	--encoder-layers 6 \
 #  --decoder-layers 6 \
 #  --max-tokens 2048 \
 #  --target-layers 3 \

#alpha0.9related
#	--encoder-layers 6 \
#  --decoder-layers 6 \
#  --max-tokens 9000 \
#  --target-layers 6 \


#alpha 0.6
#	--encoder-layers 6 \
#  --decoder-layers 6 \
#  --max-tokens 512 \
#  --target-layers 6 \

#alpha0.3
#	--encoder-layers 6 \
#  --decoder-layers 6 \
#  --max-tokens 6000 \
#  --target-layers 6 \

#	--encoder-layers 4 \
#  --decoder-layers 4 \
#  --max-tokens 4096 \
#  --target-layers 4 \

#	--encoder-layers 6 \
#  --decoder-layers 6 \
#  --max-tokens 10000 \
#  --target-layers 6 \