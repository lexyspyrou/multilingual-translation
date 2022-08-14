#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB

MODEL_DIR=checkpoints/diverse_ted8_o2m/proportional/original_rerun
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

python train.py data-bin/ted_8_diverse/ \
	  --task multilingual_translation \
	  --arch multilingual_transformer_iwslt_de_en \
	  --max-epoch 40 \
    --lang-pairs "eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor" \
	  --no-epoch-checkpoints \
	  --distributed-world-size 4 \
	  --encoder-langtok "tgt" \
	  --share-decoder-input-output-embed --share-decoders --share-encoders \
	  --dropout 0.3 --attention-dropout 0.3 --relu-dropout 0.3 --weight-decay 0.0 \
	  --left-pad-source 'True' --left-pad-target 'False' \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 \
	  --criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 2048 \
	  --update-freq 2 \
	  --seed 2 \
    --max-source-positions 150 --max-target-positions 150 \
    --save-dir $MODEL_DIR \
    --tensorboard-logdir $MODEL_DIR/tensorboard_dir.log \
    --encoder-normalize-before --decoder-normalize-before \
	  --log-interval 100 >> $MODEL_DIR/train.log 2>&1 \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend=no_c10d \
