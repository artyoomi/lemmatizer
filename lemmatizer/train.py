"""Training script for the lemmatizer."""
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .data import parse_opencorpora, build_vocabs, LemmaDataset, collate_fn, SOS, EOS, normalize
from .model import Lemmatizer


def train(xml_path: str, output_dir: str, epochs: int = 10, batch_size: int = 64,
          lr: float = 0.001, max_sentences: int | None = None, device: str | None = None):
    """Train the lemmatizer model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Parsing OpenCorpora...")
    sentences = list(parse_opencorpora(xml_path))
    if max_sentences:
        random.shuffle(sentences)
        sentences = sentences[:max_sentences]
    print(f"Loaded {len(sentences)} sentences")
    
    # Build vocabularies
    print("Building vocabularies...")
    char_vocab, word_vocab, pos_vocab = build_vocabs(sentences)
    char_vocab.save(output_path / 'char_vocab.json')
    word_vocab.save(output_path / 'word_vocab.json')
    pos_vocab.save(output_path / 'pos_vocab.json')
    print(f"Char vocab: {len(char_vocab)}, Word vocab: {len(word_vocab)}, POS vocab: {len(pos_vocab)}")
    
    # Create dataset
    dataset = LemmaDataset(sentences, char_vocab, word_vocab, pos_vocab)
    train_size = int(0.9 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn)
    
    # Create model
    model = Lemmatizer(len(char_vocab), len(word_vocab), len(pos_vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lemma_criterion = nn.CrossEntropyLoss(ignore_index=0)
    pos_criterion = nn.CrossEntropyLoss()
    
    sos_idx = char_vocab.token2idx[SOS]
    eos_idx = char_vocab.token2idx[EOS]
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (words, contexts, positions, lemmas, pos_tags) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"\rEpoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)}", end="", flush=True)
            words, contexts, positions, lemmas, pos_tags = (
                words.to(device), contexts.to(device), positions.to(device), 
                lemmas.to(device), pos_tags.to(device)
            )
            
            optimizer.zero_grad()
            outputs, pos_logits = model(words, contexts, positions, lemmas, teacher_forcing=0.5)
            lemma_loss = lemma_criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), lemmas[:, 1:].reshape(-1))
            pos_loss = pos_criterion(pos_logits, pos_tags)
            loss = lemma_loss + pos_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss, lemma_correct, pos_correct, total = 0, 0, 0, 0
        with torch.no_grad():
            for words, contexts, positions, lemmas, pos_tags in val_loader:
                words, contexts, positions, lemmas, pos_tags = (
                    words.to(device), contexts.to(device), positions.to(device),
                    lemmas.to(device), pos_tags.to(device)
                )
                outputs, pos_logits = model(words, contexts, positions, lemmas, teacher_forcing=0)
                lemma_loss = lemma_criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), lemmas[:, 1:].reshape(-1))
                pos_loss = pos_criterion(pos_logits, pos_tags)
                val_loss += (lemma_loss + pos_loss).item()
                
                # Accuracy
                preds, pos_preds = model.generate(words, contexts, positions, sos_idx, eos_idx)
                pos_correct += (pos_preds.to(device) == pos_tags).sum().item()
                for pred, target in zip(preds, lemmas):
                    pred_chars = [c for c in pred.tolist() if c not in (0, sos_idx, eos_idx)]
                    target_chars = [c for c in target.tolist() if c not in (0, sos_idx, eos_idx)]
                    if pred_chars == target_chars:
                        lemma_correct += 1
                    total += 1
        
        print()  # newline after progress
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        lemma_acc = lemma_correct / total if total > 0 else 0
        pos_acc = pos_correct / total if total > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Lemma Acc: {lemma_acc:.4f}, POS Acc: {pos_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_vocab_size': len(char_vocab),
                'word_vocab_size': len(word_vocab),
                'pos_vocab_size': len(pos_vocab),
            }, output_path / 'model.pt')
            print(f"  Saved best model")
    
    print(f"Training complete. Model saved to {output_path}")
