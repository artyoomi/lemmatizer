"""CLI interface for the lemmatizer."""
import argparse
import sys


def cli():
    parser = argparse.ArgumentParser(description='Context-aware Russian lemmatizer')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', required=True, help='Path to OpenCorpora XML file')
    train_parser.add_argument('--output', default='./model', help='Output directory for model')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--max-sentences', type=int, help='Limit sentences for faster training')
    train_parser.add_argument('--device', help='Device (cuda/cpu)')
    
    # Lemmatize command
    lemma_parser = subparsers.add_parser('lemmatize', help='Lemmatize text')
    lemma_parser.add_argument('--model', required=True, help='Path to model directory')
    lemma_parser.add_argument('--text', help='Text to lemmatize')
    lemma_parser.add_argument('--file', help='File with text to lemmatize')
    lemma_parser.add_argument('--device', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from .train import train
        train(
            xml_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_sentences=args.max_sentences,
            device=args.device
        )
    
    elif args.command == 'lemmatize':
        from .inference import LemmatizerInference
        
        lemmatizer = LemmatizerInference(args.model, device=args.device)
        
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, encoding='utf-8') as f:
                text = f.read()
        else:
            text = sys.stdin.read()
        
        for line in text.strip().split('\n'):
            if line.strip():
                results = lemmatizer.lemmatize(line)
                print(lemmatizer.format_output(results))


if __name__ == '__main__':
    cli()
