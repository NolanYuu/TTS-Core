import tacotron_cleaner.cleaners
import g2p_en


class Preprocessor:
    def __init__(
        self,
        token_list,
        unk_symbol: str = "<unk>",
    ):
        self.g2p = g2p_en.G2p()
        self.token2id = {}
        for i, t in enumerate(token_list):
            self.token2id[t] = i
        self.unk_id = self.token2id[unk_symbol]

    def tokens2ids(self, tokens):
        return [self.token2id.get(i, self.unk_id) for i in tokens]

    def __call__(self, text: str):
        text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
        tokens = self.g2p(text)
        tokens = list(filter(lambda s: s != " ", tokens))
        text = self.tokens2ids(tokens)
        return text
