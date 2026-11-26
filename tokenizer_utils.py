"""Токенизатор (BPE, учится с нуля на наших данных) и превращение диалогов в токены."""
import gzip
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA = "data/qa_ru.jsonl.gz"
TOKENIZER_PATH = "model/tokenizer.json"
SPECIAL = ["<|pad|>", "<|user|>", "<|bot|>", "<|end|>"]


def iter_dialogs(path=DATA):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["messages"]


def train_tokenizer(vocab_size=8192, path=TOKENIZER_PATH):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    texts = (m["content"] for d in iter_dialogs() for m in d)
    tok.train_from_iterator(texts, trainer=trainer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok.save(path)
    return tok


def load_tokenizer(path=TOKENIZER_PATH):
    return Tokenizer.from_file(path)


class ChatFormat:
    """<|user|>вопрос<|end|><|bot|>ответ<|end|><|user|>..."""

    def __init__(self, tok):
        self.tok = tok
        self.user = tok.token_to_id("<|user|>")
        self.bot = tok.token_to_id("<|bot|>")
        self.end = tok.token_to_id("<|end|>")

    def encode_message(self, role, text):
        head = self.user if role == "user" else self.bot
        return [head] + self.tok.encode(text).ids + [self.end]

    def encode_dialog(self, messages):
        ids = []
        for m in messages:
            ids += self.encode_message(m["role"], m["content"])
        return ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)
