# Universal tokenizers

## Build wordpiece vocab all file .txt and .tok
```bash
python build_wordpiece_vocab.py \
    --corpus_dir data \
    --output_dir outputs
```

## Build SentencePiece vocab

* Installation

```bash
pip install sentencepiece
```
* Update for SentencePiece
SentencePiece work much better if use --split_by_whitespace=false, ignore whitespace and go for high frequency characters/words, and give interesting result
```bash
['▁Hà▁Nội', '▁vừa', '▁thông▁tin', '▁trường▁hợp', '▁mắc', '▁Covid']
```

