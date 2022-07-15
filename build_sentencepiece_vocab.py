import sentencepiece as spm

# spm.SentencePieceTrainer.train(
#     '--input=data/news_title.txt --model_prefix=data/spm --max_sentencepiece_length=20 --split_digits --hard_vocab_limit=false --model_type=bpe --character_coverage=0.99 --vocab_size=10000 --split_by_whitespace=false --normalization_rule_name=nfkc'
# )

# spm.SentencePieceTrainer.train(
#     '--input=corpus-title.txt --model_prefix=data/spm --model_type=unigram --character_coverage=0.999 --vocab_size=64000 --split_by_whitespace=false --normalization_rule_name=nfkc'
# )

# makes segmenter instance and load the model file
sp = spm.SentencePieceProcessor()
sp.load('data/spm.model')

# encode: text => id
print(sp.encode_as_pieces('Hà Nội vừa thông tin trường hợp mắc Covid'))
print(sp.encode_as_pieces('Chiến thuật đối phó của pháo binh Ukraine với hỏa lực vượt trội từ Nga'))
# print(sp.nbest_encode_as_pieces('tôi yêu Hà Nội', 10))
print(sp.encode_as_ids('tôi yêu Hà Nội'))
print(sp.piece_to_id('<s>'))  # 3
print(sp.piece_to_id('</s>'))  # 4

# returns vocab size
print(sp.get_piece_size())

