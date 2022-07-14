# -*- coding: utf-8 -*-
from pathlib import Path
import torch
from tokenizers import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from transformers import ElectraTokenizer, BertTokenizer, ElectraTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFC
# from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


def create_vocab(args):
    paths = [str(x) for x in Path(str(args.corpus_dir + "/")).glob("**/*.tok")]
    paths.extend([str(x) for x in Path(str(args.corpus_dir + "/")).glob("**/*.txt")])
    print(paths)

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=args.lowercase,
    )
    tokenizer._tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(files=paths,
                    vocab_size=args.vocab_size,
                    min_frequency=300,
                    limit_alphabet=185,
                    special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"],
                    wordpieces_prefix="##")

    tokenizer.save_model(args.output_dir)


class WordPieceTokenizer(BertWordPieceTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, clean_text=False,
                         handle_chinese_chars=False,
                         strip_accents=False,
                         lowercase=True, **kwargs)
        self._tokenizer.pre_tokenizer = Whitespace()

    def tokenize(self, text, add_special_token=False):
        return self.encode(text, add_special_tokens=add_special_token).tokens


if __name__ == "__main__":
    tokenizerBW = WordPieceTokenizer(
        "../prev_trained_model/vocab.txt"
    )

    text = "Thủ_tướng Nguyễn_Xuân_Phúc điện_đàm với Tổng_thống Hoa_Kỳ Donald_Trump .\nCùng đi xem countdown đê."
#     text = """
#      Nhập cảnh trái phép vào Việt Nam, 5 người Trung Quốc từ Hải Phòng đi xe khách đến TPHCM mới bị phát hiện. Lực lượng y tế đã lấy mẫu xét nghiệm, chuyển nhóm người trên đến khu cách ly tập trung Củ Chi.
#
# Trung tâm Y tế quận Thủ Đức (TPHCM) vừa có báo cáo nhanh về việc phát hiện một nhóm người Trung Quốc nhập cảnh trái phép vào Việt Nam được phát hiện trên địa bàn. Theo đó, vào lúc 1h45 ngày 28/12/2020 tại địa chỉ số 171, quốc lộ 1A, khu phố 3, phường Bình Chiểu, Công an Giao thông Bình Triệu tiến hành kiểm tra và phát hiện trên xe khách biển kiểm soát 51B- 09750 có 5 khách là người Trung Quốc.
#     """
    # tokenized_sequenceBW = tokenizerBW.encode(text)
    # token = tokenized_sequenceBW.tokens
    # print(tokenized_sequenceBW)
    # print(tokenized_sequenceBW.tokens)
    # print(tokenized_sequenceBW.ids)
    # print([tokenizerBW.token_to_id(t) for t in token])
    # print(tokenizerBW.tokenize(text))
    # print(tokenizerBW.token_to_id('[PAD]'))
    # print(tokenizerBW.get_vocab().keys())

    # print("electra fast tokenizer" + "="*20)
    # tokenizer = ElectraTokenizerFast(vocab_file="../prev_trained_model/vocab.txt",
    #                                  do_basic_tokenize=False,
    #                                  strip_accents=False
    #                                  )
    # encoding = tokenizer.encode_plus(text.lower(),
    #                                  return_overflowing_tokens=True,
    #                                  max_length=5)
    # print(encoding)
    # features = []
    # for overflowing_encoding in encoding.encodings[0].overflowing:
    #     features.append(torch.tensor(overflowing_encoding.ids))
    # print(features)

    print("Using electra tokenizer" + "=" * 20)
    tokenizer = ElectraTokenizer(vocab_file="../prev_trained_model/vocab.txt",
                                 do_basic_tokenize=False
                                 )

    features = []
    # features = torch.tensor()
    # encoding = tokenizer.batch_encode_plus(text.lower().split("\n"),
    #                                        return_overflowing_tokens=True,
    #                                        max_length=15,
    #                                        padding=True)
    # print(encoding)
    block_size = 5

    # tokenizer._tokenizer.pre_tokenizer = Whitespace()
    print(tokenizer.tokenize(text.lower()))
    print(tokenizer.encode(text.lower()))
    print(tokenizer.max_len)
