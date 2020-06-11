import os
import sentencepiece as spm
import sentencepiece_pb2


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_path, "spiece.model"))
    
    def encode(self, sentence):
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            offsets.append((piece.begin, piece.end))
        return tokens, offsets
    
spt = SentencePieceTokenizer("xlnet_model/")
tokens, offsets = spt.encode("hi, how are you?")
print(tokens)
print(offsets)


