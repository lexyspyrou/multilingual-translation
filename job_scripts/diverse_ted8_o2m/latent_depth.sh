#lang_pairs_str="eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor"
lang_pairs_str="eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor"
#  --arch multilingual_transformer_iwslt_de_en \

# Code adapted to work on a small GPU: In specific, --max-tokens 4096, --encoder-layers 12, --decoder-layers 24, --target-layers 12

MODEL_DIR=checkpoints/diverse_ted8_o2m/latent_depth/big/
mkdir -p $MODEL_DIR
export PYTHONPATH="$(pwd)"

echo 'slurm id '$SLURM_JOB_ID >> $MODEL_DIR/train.log

fairseq-train data-bin/ted_8_diverse/ \
  --user-dir examples/latent_depth/latent_depth_src \
  --lang-pairs "${lang_pairs_str}" \
  --arch latent_multilingual_transformer \
  --task multilingual_translation_latent_depth \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --share-encoders \
  --share-decoders \
	--max-epoch 40 \
  --decoder-langtok \
  --share-decoder-input-output-embed \
  --dropout 0.3 --attention-dropout 0.3 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --stop-min-lr 1e-9 --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --max-tokens 4096 --update-freq 1  \
  --lr 0.0015 \
  --clip-norm 1.0 \
  --seed 2 \
  --ddp-backend=legacy_ddp \
  --encoder-layers 12 \
  --decoder-layers 24 \
  --decoder-latent-layer \
  --sparsity-weight 0.1 \
  --anneal-updates 5000 \
  --soft-update 500  \
  --target-layers 12 \
  --save-dir $MODEL_DIR \
	--log-interval 100 >> $MODEL_DIR/train.log 2>&1 \
  --tensorboard-logdir $MODEL_DIR/tensorboard_dir.log \
  --no-epoch-checkpoints \
  --distributed-world-size 4 \
  --share-weight 0.1 \
  --encoder-embed-dim 512 \
  --decoder-embed-dim 512 \
  --encoder-ffn-embed-dim 1024 \
  --decoder-ffn-embed-dim 1024 \
