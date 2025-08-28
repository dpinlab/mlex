#!/usr/bin/env python3
"""
Command-line interface for MLEX (Money Laundering Expert System)
"""

import argparse
import sys
from pathlib import Path

from mlex.models import GRU, LSTM, RNN
from mlex.utils import DataReader, FeatureStratifiedSplit
from mlex.evaluation import StandardEvaluator, F1MaxThresholdStrategy


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Money Laundering Expert System (MLEX) - Financial Fraud Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlex train --data data.csv --target fraud_label --model gru
  mlex evaluate --model model.pkl --data test.csv --target fraud_label
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data', required=True, help='Path to training data CSV')
    train_parser.add_argument('--target', required=True, help='Target column name')
    train_parser.add_argument('--model', choices=['gru', 'lstm', 'rnn'], default='gru', help='Model type')
    train_parser.add_argument('--output', default='model.pkl', help='Output model file')
    train_parser.add_argument('--val-split', type=float, default=0.3, help='Validation split proportion')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--hidden-size', type=int, default=64, help='Hidden layer size')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    eval_parser.add_argument('--data', required=True, help='Path to test data CSV')
    eval_parser.add_argument('--target', required=True, help='Target column name')
    eval_parser.add_argument('--output', default='evaluation.json', help='Output evaluation file')
    
    # Version command
    subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if args.command == 'version':
        from mlex import __version__
        print(f"MLEX version {__version__}")
        return
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    else:
        parser.print_help()
        sys.exit(1)


def train_model(args):
    """Train a model with the given parameters"""
    print(f"Loading data from {args.data}...")
    
    # Load data
    reader = DataReader(args.data, target_columns=[args.target])
    X = reader.fit_transform()
    y = reader.get_target()
    
    # Split data
    print("Splitting data...")
    splitter = FeatureStratifiedSplit(test_proportion=args.test_split)
    splitter.fit(X, y)
    X_train, y_train, X_test, y_test = splitter.transform(X, y)
    
    # Create model
    model_classes = {'gru': GRU, 'lstm': LSTM, 'rnn': RNN}
    model_class = model_classes[args.model]
    
    print(f"Training {args.model.upper()} model...")
    model = model_class(
        target_column=args.target,
        validation_data=(X_test, y_test),
        hidden_size=args.hidden_size,
        epochs=args.epochs
    )
    
    model.fit(X_train, y_train)
    
    # Save model
    import pickle
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {args.output}")


def evaluate_model(args):
    """Evaluate a trained model"""
    print(f"Loading model from {args.model}...")
    
    # Load model
    import pickle
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    print(f"Loading test data from {args.data}...")
    reader = DataReader(args.data, target_columns=[args.target])
    X_test = reader.fit_transform()
    y_test = reader.get_target()
    
    # Evaluate
    print("Evaluating model...")
    scores = model.score_samples(X_test)
    
    evaluator = StandardEvaluator("model_evaluation", F1MaxThresholdStrategy())
    evaluator.evaluate(y_test, [], scores)
    
    # Save results
    evaluator.save(args.output)
    print(f"Evaluation results saved to {args.output}")
    print("\nEvaluation Summary:")
    print(evaluator.summary())


if __name__ == '__main__':
    main() 