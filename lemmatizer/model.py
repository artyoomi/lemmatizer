"""Context-aware lemmatization model: BiLSTM encoder + attention decoder + POS classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """BiLSTM encoder for word characters + context."""

    def __init__(self, char_vocab_size: int, word_vocab_size: int,
                 char_emb_dim: int = 64, word_emb_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.ctx_lstm = nn.LSTM(word_emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, word_chars, context, positions):
        # Encode word characters
        char_emb = self.char_emb(word_chars)
        char_out, (h_char, c_char) = self.char_lstm(char_emb)
        
        # Encode context
        ctx_emb = self.word_emb(context)
        ctx_out, _ = self.ctx_lstm(ctx_emb)
        
        # Get context vector at word position
        batch_idx = torch.arange(ctx_out.size(0), device=ctx_out.device)
        positions = positions.clamp(0, ctx_out.size(1) - 1)
        ctx_vec = ctx_out[batch_idx, positions]  # (batch, hidden)
        
        # Combine hidden states
        h = torch.cat([h_char[0], h_char[1]], dim=1).unsqueeze(0)
        c = torch.cat([c_char[0], c_char[1]], dim=1).unsqueeze(0)
        
        return char_out, ctx_vec, (h, c)


class Decoder(nn.Module):
    """LSTM decoder with attention for lemma generation."""

    def __init__(self, char_vocab_size: int, emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(char_vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTMCell(emb_dim + hidden_dim * 2, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, char_vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, input_char, hidden, cell, encoder_out, ctx_vec):
        emb = self.emb(input_char)  # (batch, emb_dim)
        
        # Attention over encoder outputs
        attn_weights = F.softmax(
            torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2), dim=1
        )
        attn_vec = torch.bmm(attn_weights.unsqueeze(1), encoder_out).squeeze(1)
        
        # Combine with context
        lstm_input = torch.cat([emb, attn_vec, ctx_vec], dim=1)
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        
        logits = self.out(hidden)
        return logits, hidden, cell


class Lemmatizer(nn.Module):
    """Full lemmatization model with POS tagging."""

    def __init__(self, char_vocab_size: int, word_vocab_size: int, pos_vocab_size: int,
                 char_emb_dim: int = 64, word_emb_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.encoder = Encoder(char_vocab_size, word_vocab_size, char_emb_dim, word_emb_dim, hidden_dim)
        self.decoder = Decoder(char_vocab_size, char_emb_dim, hidden_dim)
        self.pos_classifier = nn.Linear(hidden_dim * 2, pos_vocab_size)  # from char + ctx
        self.hidden_dim = hidden_dim

    def forward(self, word_chars, context, positions, target_lemma, teacher_forcing=0.5):
        batch_size = word_chars.size(0)
        max_len = target_lemma.size(1)
        vocab_size = self.decoder.out.out_features
        
        encoder_out, ctx_vec, (h, c) = self.encoder(word_chars, context, positions)
        hidden, cell = h.squeeze(0), c.squeeze(0)
        
        # POS classification from combined representation
        pos_logits = self.pos_classifier(torch.cat([hidden, ctx_vec], dim=1))
        
        outputs = torch.zeros(batch_size, max_len, vocab_size, device=word_chars.device)
        input_char = target_lemma[:, 0]  # SOS token
        
        for t in range(1, max_len):
            logits, hidden, cell = self.decoder(input_char, hidden, cell, encoder_out, ctx_vec)
            outputs[:, t] = logits
            
            if torch.rand(1).item() < teacher_forcing:
                input_char = target_lemma[:, t]
            else:
                input_char = logits.argmax(dim=1)
        
        return outputs, pos_logits

    def generate(self, word_chars, context, positions, sos_idx: int, eos_idx: int, max_len: int = 50):
        """Generate lemma and POS for inference."""
        self.eval()
        with torch.no_grad():
            encoder_out, ctx_vec, (h, c) = self.encoder(word_chars, context, positions)
            hidden, cell = h.squeeze(0), c.squeeze(0)
            
            # POS prediction
            pos_logits = self.pos_classifier(torch.cat([hidden, ctx_vec], dim=1))
            pos_preds = pos_logits.argmax(dim=1)
            
            batch_size = word_chars.size(0)
            input_char = torch.full((batch_size,), sos_idx, device=word_chars.device)
            result = [input_char.tolist()]
            
            for _ in range(max_len):
                logits, hidden, cell = self.decoder(input_char, hidden, cell, encoder_out, ctx_vec)
                input_char = logits.argmax(dim=1)
                result.append(input_char.tolist())
                if (input_char == eos_idx).all():
                    break
            
            return torch.tensor(result).T, pos_preds  # (batch, seq_len), (batch,)
