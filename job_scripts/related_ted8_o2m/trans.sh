#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB
#     	  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
	     #   --decoder-langtok \ when o2m setting // encoder-langtok "tgt" for m2o
# DURING INFERENCE THE datasize-t argumenmt of multilingual_translation.py creates an issue (arg conflict) so i manually remove it

	  #  ADD THIS ONLY TO TEMPERATURE MODEL      --datasize-t 5 \

OUTDIR=$1
echo $OUTDIR
python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --encoder-langtok tgt \
	        --skip-invalid-size-inputs-valid-test \
    	    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang aze \
          --beam 5   > "$OUTDIR"/test_engaze.log

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --encoder-langtok tgt \
	        --skip-invalid-size-inputs-valid-test \
    	    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang tur \
          --beam 5   > "$OUTDIR"/test_engtur.log

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --encoder-langtok tgt \
	        --skip-invalid-size-inputs-valid-test \
      	  --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang bel \
          --beam 5  > "$OUTDIR"/test_engbel.log

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --encoder-langtok tgt \
	        --skip-invalid-size-inputs-valid-test \
    	    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang rus \
          --beam 5   > "$OUTDIR"/test_engrus.log

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --encoder-langtok tgt \
	        --skip-invalid-size-inputs-valid-test \
    	    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang glg \
          --beam 5   > "$OUTDIR"/test_engglg.log

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --encoder-langtok tgt \
	        --skip-invalid-size-inputs-valid-test \
    	    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang por \
          --beam 5   > "$OUTDIR"/test_engpor.log

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --encoder-langtok tgt \
	        --skip-invalid-size-inputs-valid-test \
    	    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang slk \
          --beam 5   > "$OUTDIR"/test_engslk.log

python fairseq_cli/generate.py data-bin/ted_8_related/ \
          --task multilingual_translation \
          --gen-subset test \
          --path "$OUTDIR"/checkpoint_best.pt \
          --batch-size 32 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --sacrebleu \
          --encoder-langtok tgt \
	        --skip-invalid-size-inputs-valid-test \
    	    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
          --source-lang eng --target-lang ces \
          --beam 5   > "$OUTDIR"/test_engces.log

