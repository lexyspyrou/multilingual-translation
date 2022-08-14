#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB
#CHANGE THE INFERENCE COMMAND TO SHOW WHERE checkpoint_best.pt IS SAVED
# BE CAREFUL WHERE THE TRANSLATIONS ARE SAVED
lang_pairs_str="eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor"

src_lang="eng"

OUTDIR=$1
echo $OUTDIR
python fairseq_cli/generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "bos" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engbos.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "mar" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engmar.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "hin" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_enghin.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "mkd" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engmkd.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "ell" \
          --sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engell.log \
          --decoder-langtok \
          --batch-size 32 \


python fairseq_cli/generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "bul" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engbul.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "fra" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engfra.log \
          --decoder-langtok \
          --batch-size 32 \
#
python fairseq_cli/generate.py data-bin/ted_8_diverse/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "kor" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engkor.log \
          --decoder-langtok \
          --batch-size 32 \
