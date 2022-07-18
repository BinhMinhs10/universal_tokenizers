import sentencepiece as spm

spm.SentencePieceTrainer.train(
    '--input=data/news_title.txt --model_prefix=data/spm --user_defined_symbols=<mask>,<pad> --max_sentencepiece_length=20 --split_digits --hard_vocab_limit=false --model_type=bpe --character_coverage=0.99 --vocab_size=10000 --split_by_whitespace=false --normalization_rule_name=nfkc'
)

# spm.SentencePieceTrainer.train(
#     '--input=corpus-title.txt --model_prefix=data/spm --model_type=unigram --character_coverage=0.999 --vocab_size=64000 --split_by_whitespace=false --normalization_rule_name=nfkc'
# )

# makes segmenter instance and load the model file
sp = spm.SentencePieceProcessor()
sp.load('data/spm.model')
# sp.load('data/sentencepiece.bpe.model')

# encode: text => id
print(sp.encode_as_pieces('Hà Nội vừa thông tin trường hợp mắc Covid'))
print(sp.encode_as_pieces('Chiến thuật đối phó của pháo binh Ukraine với hỏa lực vượt trội từ Nga'))
# print(sp.nbest_encode_as_pieces('tôi yêu Hà Nội', 10))
print(sp.encode_as_ids('tôi yêu Hà Nội'))
print(sp.piece_to_id('<s>'))  # 3
print(sp.piece_to_id('</s>'))  # 4
print(sp.piece_to_id('<unk>'))
print(sp.piece_to_id('<mask>'))
print(sp.piece_to_id('<pad>'))

print(sp.id_to_piece(sp.bos_id()), ' is bos=', sp.bos_id())
print(sp.id_to_piece(sp.eos_id()), ' is eos=', sp.eos_id())
print(sp.id_to_piece(sp.unk_id()), ' is unk=', sp.unk_id())
print(sp.id_to_piece(3), ' is mask=', sp.piece_to_id('<mask>'))
print('pad=', sp.pad_id())

# returns vocab size
print(sp.get_piece_size())

