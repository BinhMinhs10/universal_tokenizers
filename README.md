# Universal tokenizers
make a purely end-to-end system that does not depend on language-specific preprocessing

## Build wordpiece vocab all file .txt and .tok
```bash
python build_wordpiece_vocab.py \
    --corpus_dir data \
    --output_dir outputs
```

## Build SentencePiece vocab
implement **subword units** (e.g BPE) and **unigram language model**


### Installation

```bash
pip install sentencepiece
```
### Update for SentencePiece
SentencePiece work much better if use --split_by_whitespace=false, ignore whitespace and go for high frequency characters/words, and give interesting result
```bash
['▁Hà▁Nội', '▁vừa', '▁thông▁tin', '▁trường▁hợp', '▁mắc', '▁Covid']
```

### Usage instruction
```bash
python build_sentencepiece_vocab.py
```

* `--input`: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor. By default, SentencePiece normalizes the input with Unicode NFKC. You can pass a comma-separated list of files.
* `--model_prefix`: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
* `--vocab_size`: vocabulary size, e.g., 8000, 16000, or 32000
* `--character_coverage`: amount of characters covered by the model, good defaults are: 0.9995 for languages with rich character set like Japanese or Chinese and 1.0 for other languages with small character set.
* ``--model_type`: model type. Choose from `unigram` (default), `bpe`, `char`, or `word`. The input sentence must be pretokenized when using `word` type.
* `--user_defined_symbols`: always treated as one token in any context


## Resource:
* [Sentencepiece python module example](https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb#scrollTo=Lf5Fs_pPIKif)