#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB
#CHANGE THE INFERENCE COMMAND TO SHOW WHERE checkpoint_best.pt IS SAVED
# BE CAREFUL WHERE THE TRANSLATIONS ARE SAVED
lang_pairs_str="eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces"

src_lang="eng"

OUTDIR=$1
echo $OUTDIR
python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "aze" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engaze.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "tur" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engtur.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "bel" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engbel.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "rus" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engrus.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "glg" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engglg.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "por" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engpor.log \
          --decoder-langtok \
          --batch-size 32 \

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "slk" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engslk.log \
          --decoder-langtok \
          --batch-size 32 \
#
python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation_latent_depth \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --decoder-latent-layer \
          --lang-pairs "${lang_pairs_str}" \
          -s ${src_lang} -t "ces" \
          --scoring sacrebleu \
          --remove-bpe 'sentencepiece' \
          --lenpen 1.0 \
          --beam 5   > "$OUTDIR"/test_engces.log \
          --decoder-langtok \
          --batch-size 32 \
