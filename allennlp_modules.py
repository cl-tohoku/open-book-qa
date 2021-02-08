import os
from typing import List

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("mecab")
class MeCabTokenizer(Tokenizer):
    def __init__(self) -> None:
        import fugashi
        import ipadic
        dic_dir = ipadic.DICDIR
        mecabrc = os.path.join(dic_dir, "mecabrc")
        mecab_option = '-d "{}" -r "{}" '.format(dic_dir, mecabrc)
        self.mecab = fugashi.GenericTagger(mecab_option)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = []
        cursor = 0
        for word in self.mecab(text):
            token = word.surface
            start = text.index(token, cursor)
            end = start + len(token)
            tokens.append(Token(text=token, idx=start, idx_end=end))
            cursor = end

        return tokens
