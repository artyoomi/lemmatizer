"""Inference module for lemmatizing text."""
import re
from pathlib import Path
import torch

from .data import Vocab, SOS, EOS, normalize
from .model import Lemmatizer


class LemmatizerInference:
    """Inference wrapper for the trained lemmatizer."""

    def __init__(self, model_dir: str, device: str | None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = Path(model_dir)
        
        # Load vocabularies
        self.char_vocab = Vocab.load(model_path / 'char_vocab.json')
        self.word_vocab = Vocab.load(model_path / 'word_vocab.json')
        self.pos_vocab = Vocab.load(model_path / 'pos_vocab.json')
        
        # Load model
        checkpoint = torch.load(model_path / 'model.pt', map_location=self.device, weights_only=True)
        self.model = Lemmatizer(
            checkpoint['char_vocab_size'],
            checkpoint['word_vocab_size'],
            checkpoint['pos_vocab_size']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.sos_idx = self.char_vocab.token2idx[SOS]
        self.eos_idx = self.char_vocab.token2idx[EOS]
        self.unk_idx = self.char_vocab.token2idx['<UNK>']

    def _tokenize(self, text: str) -> list[str]:
        """Split text into words, removing punctuation."""
        return re.findall(r'[а-яёА-ЯЁa-zA-Z]+', text)

    def _is_unknown(self, word: str) -> bool:
        """Check if word contains unknown characters."""
        normalized = normalize(word)
        return any(c not in self.char_vocab.token2idx for c in normalized)

    def lemmatize(self, text: str) -> list[tuple[str, str, str]]:
        """Lemmatize text, returning list of (word, lemma, pos) tuples."""
        words = self._tokenize(text)
        if not words:
            return []
        
        words_normalized = [normalize(w) for w in words]
        ctx_ids = self.word_vocab.encode(words_normalized)
        ctx_tensor = torch.tensor([ctx_ids], device=self.device)
        
        results = []
        for i, (orig, norm) in enumerate(zip(words, words_normalized)):
            # Handle unknown words
            if self._is_unknown(orig):
                results.append((orig, norm, 'NI'))
                continue
            
            word_chars = [SOS] + list(norm) + [EOS]
            word_tensor = torch.tensor([self.char_vocab.encode(word_chars)], device=self.device)
            pos_tensor = torch.tensor([i], device=self.device)
            
            with torch.no_grad():
                output, pos_pred = self.model.generate(word_tensor, ctx_tensor, pos_tensor, 
                                                        self.sos_idx, self.eos_idx)
            
            lemma_ids = output[0].tolist()
            lemma_chars = [c for c in self.char_vocab.decode(lemma_ids) 
                          if c not in (SOS, EOS, '<PAD>', '<UNK>')]
            lemma = ''.join(lemma_chars)
            
            pos = self.pos_vocab.idx2token.get(pos_pred[0].item(), 'NI')
            if pos in ('<PAD>', '<UNK>', '<SOS>', '<EOS>'):
                pos = 'NI'
            
            results.append((orig, lemma, pos))
        
        return results

    def format_output(self, results: list[tuple[str, str, str]]) -> str:
        """Format results as Word{lemma=POS} string."""
        return ' '.join(f"{word}{{{lemma}={pos}}}" for word, lemma, pos in results)
