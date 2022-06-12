import argparse
import sys

import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args = parser.parse_args()

spsrc = spm.SentencePieceProcessor()
spsrc.Load(args.model)
for line in sys.stdin:
    print(" ".join(spsrc.EncodeAsPieces(line.strip())))
