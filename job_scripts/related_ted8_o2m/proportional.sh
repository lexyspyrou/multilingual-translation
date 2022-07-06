#!/bin/bash
#    --encoder-layers 4 \
#    --decoder-layers 4 \
#    --encoder-embed-dim 256 \
#    --decoder-embed-dim 256 \
#    --encoder-ffn-embed-dim 512 \
#    --decoder-ffn-embed-dim 512 \
MODEL_DIR=checkpoints/related_ted8_o2m/proportional/
mkdir -p $MODEL_DIR

export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

python train.py data-bin/ted_8_related/ \
	  --task multilingual_translation \
	  --arch multilingual_transformer_iwslt_de_en \
	  --max-epoch 40  \
    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
    --no-epoch-checkpoints \
	  --distributed-world-size 4 \
	  --encoder-langtok "tgt" \
	  --share-decoder-input-output-embed --share-decoders --share-encoders \
	  --dropout 0.3 --attention-dropout 0.3 --relu-dropout 0.3 --weight-decay 0.0 \
	  --left-pad-source 'True' --left-pad-target 'False' \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 \
	  --criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 2000 \
	  --update-freq 2 \
	  --seed 2 \
    --max-source-positions 150 --max-target-positions 150 \
    --save-dir $MODEL_DIR \
    --encoder-normalize-before --decoder-normalize-before \
	  --log-interval 100 >> $MODEL_DIR/train.log 2>&1 \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend=no_c10d