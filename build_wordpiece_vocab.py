import argparse

from processors.build_wordpiece_vocab import create_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus_dir", required=True,
                        help="Location of folder pre-training all text files .txt or .tok.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write out the vocabulary file.")
    parser.add_argument("--max-seq-length", default=128, type=int,
                        help="Number of tokens per example.")
    parser.add_argument("--vocab_size", default=65000, type=int,
                        help="Number of vocab size")
    parser.add_argument("--lowercase", default=False, type=bool,
                        help="Lower case input text.")
    parser.add_argument("--blanks-separate-docs", default=True, type=bool,
                        help="Whether blank lines indicate document boundaries.")
    args = parser.parse_args()

    create_vocab(args)


if __name__ == "__main__":
    main()
