#!/bin/sh
set -eu
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode sparse_attention_short_paper.tex
pdflatex -interaction=nonstopmode sparse_attention_short_paper.tex
cp sparse_attention_short_paper.pdf ..
rm -f sparse_attention_short_paper.pdf
