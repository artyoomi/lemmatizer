"""Data loading and vocabulary management for OpenCorpora dataset."""
import json
from pathlib import Path
from xml.etree import ElementTree
from typing import Iterator
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD, UNK, SOS, EOS = '<PAD>', '<UNK>', '<SOS>', '<EOS>'
SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]

# Map OpenCorpora POS to simplified tags
POS_MAP = {
    'NOUN': 'S', 'ADJF': 'A', 'ADJS': 'A', 'COMP': 'A',
    'VERB': 'V', 'INFN': 'V', 'PRTF': 'V', 'PRTS': 'V', 'GRND': 'V',
    'NUMR': 'NI', 'ADVB': 'ADV', 'NPRO': 'NI',
    'PRED': 'ADV', 'PREP': 'PR', 'CONJ': 'CONJ', 'PRCL': 'ADV', 'INTJ': 'ADV',
}


def normalize(text: str) -> str:
    """Normalize text: lowercase and ё->е."""
    return text.lower().replace('ё', 'е')


@dataclass
class Vocab:
    """Character/token vocabulary."""
    token2idx: dict[str, int]
    idx2token: dict[int, str]

    def __len__(self):
        return len(self.token2idx)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token2idx.get(t, self.token2idx[UNK]) for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx2token.get(i, UNK) for i in indices]

    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token2idx, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'Vocab':
        with open(path, encoding='utf-8') as f:
            token2idx = json.load(f)
        return cls(token2idx, {v: k for k, v in token2idx.items()})

    @classmethod
    def build(cls, tokens: set[str]) -> 'Vocab':
        token2idx = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        for t in sorted(tokens):
            if t not in token2idx:
                token2idx[t] = len(token2idx)
        return cls(token2idx, {v: k for k, v in token2idx.items()})


def parse_opencorpora(xml_path: str) -> Iterator[list[tuple[str, str, str]]]:
    """Parse OpenCorpora XML, yield sentences as [(word, lemma, pos), ...]."""
    for event, elem in ElementTree.iterparse(xml_path, events=['end']):
        if elem.tag != 'sentence':
            continue
        tokens = []
        for token_tag in elem.findall('.//token'):
            text = token_tag.get('text', '')
            l_tag = token_tag.find('.//l')
            if l_tag is None:
                continue
            lemma = l_tag.get('t', '')
            pos_tag = l_tag.find('.//g')
            if pos_tag is None:
                continue
            pos = pos_tag.get('v')
            if pos == 'PNCT':
                continue
            pos_simple = POS_MAP.get(pos, 'NI')
            if text and lemma:
                tokens.append((normalize(text), normalize(lemma), pos_simple))
        if tokens:
            yield tokens
        elem.clear()


class LemmaDataset(Dataset):
    """Dataset for context-aware lemmatization with POS tagging."""

    def __init__(self, sentences: list[list[tuple[str, str, str]]], 
                 char_vocab: Vocab, word_vocab: Vocab, pos_vocab: Vocab):
        self.data = []
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        
        for sent in sentences:
            words = [w for w, _, _ in sent]
            ctx_ids = word_vocab.encode(words)
            for i, (word, lemma, pos) in enumerate(sent):
                word_chars = [SOS] + list(word) + [EOS]
                lemma_chars = [SOS] + list(lemma) + [EOS]
                self.data.append((
                    char_vocab.encode(word_chars),
                    ctx_ids,
                    i,
                    char_vocab.encode(lemma_chars),
                    pos_vocab.token2idx.get(pos, pos_vocab.token2idx[UNK])
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word, ctx, pos_idx, lemma, pos_tag = self.data[idx]
        return (torch.tensor(word), torch.tensor(ctx), pos_idx, torch.tensor(lemma), pos_tag)


def collate_fn(batch):
    """Collate batch with padding."""
    words, contexts, positions, lemmas, pos_tags = zip(*batch)
    
    words_padded = pad_sequence(words, batch_first=True, padding_value=0)
    contexts_padded = pad_sequence(contexts, batch_first=True, padding_value=0)
    lemmas_padded = pad_sequence(lemmas, batch_first=True, padding_value=0)
    positions = torch.tensor(positions)
    pos_tags = torch.tensor(pos_tags)
    
    return words_padded, contexts_padded, positions, lemmas_padded, pos_tags


def build_vocabs(sentences: list[list[tuple[str, str, str]]]) -> tuple[Vocab, Vocab, Vocab]:
    """Build character, word, and POS vocabularies from sentences."""
    chars, words, pos_tags = set(), set(), set()
    for sent in sentences:
        for word, lemma, pos in sent:
            chars.update(word)
            chars.update(lemma)
            words.add(word)
            pos_tags.add(pos)
    return Vocab.build(chars), Vocab.build(words), Vocab.build(pos_tags)
