#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB
#     	  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
lang_pairs_str="eng-bel,eng-rus"
src_lang="eng"

OUTDIR=$1
echo $OUTDIR
#python generate.py data-bin/ted_8_related/ \
#          --task multilingual_translation \
#          --gen-subset test \
#          --path "$OUTDIR"/checkpoint_best.pt \
#          --batch-size 32 \
#          --lenpen 1.0 \
#          --remove-bpe sentencepiece \
#	  --sacrebleu \
#    	  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
#          --source-lang eng --target-lang aze \
#          --beam 5   > "$OUTDIR"/test_engaze.log
#
#python generate.py data-bin/ted_8_related/ \
#          --task multilingual_translation \
#          --gen-subset test \
#          --path "$OUTDIR"/checkpoint_best.pt \
#          --batch-size 32 \
#          --lenpen 1.0 \
#          --remove-bpe sentencepiece \
#	  --sacrebleu \
#    	  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
#          --source-lang eng --target-lang tur \
#          --beam 5   > "$OUTDIR"/test_engtur.log

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
          --beam 5  \
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
          --beam 5  \
          --decoder-langtok \
          --batch-size 32 \
#
#python generate.py data-bin/ted_8_related/ \
#          --task multilingual_translation \
#          --gen-subset test \
#          --path "$OUTDIR"/checkpoint_best.pt \
#          --batch-size 32 \
#          --lenpen 1.0 \
#          --remove-bpe sentencepiece \
#	  --sacrebleu \
#    	  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
#          --source-lang eng --target-lang glg \
#          --beam 5   > "$OUTDIR"/test_engglg.log
#
#python generate.py data-bin/ted_8_related/ \
#          --task multilingual_translation \
#          --gen-subset test \
#          --path "$OUTDIR"/checkpoint_best.pt \
#          --batch-size 32 \
#          --lenpen 1.0 \
#          --remove-bpe sentencepiece \
#	  --sacrebleu \
#    	  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
#          --source-lang eng --target-lang por \
#          --beam 5   > "$OUTDIR"/test_engpor.log
#
#python generate.py data-bin/ted_8_related/ \
#          --task multilingual_translation \
#          --gen-subset test \
#          --path "$OUTDIR"/checkpoint_best.pt \
#          --batch-size 32 \
#          --lenpen 1.0 \
#          --remove-bpe sentencepiece \
#	  --sacrebleu \
#    	  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
#          --source-lang eng --target-lang slk \
#          --beam 5   > "$OUTDIR"/test_engslk.log
#
#python generate.py data-bin/ted_8_related/ \
#          --task multilingual_translation \
#          --gen-subset test \
#          --path "$OUTDIR"/checkpoint_best.pt \
#          --batch-size 32 \
#          --lenpen 1.0 \
#          --remove-bpe sentencepiece \
#	  --sacrebleu \
#    	  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
#          --source-lang eng --target-lang ces \
#          --beam 5   > "$OUTDIR"/test_engces.log
#
