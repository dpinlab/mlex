# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of MLEX framework
- GRU, LSTM, and RNN neural network models
- Comprehensive evaluation framework with multiple metrics
- Data preprocessing and feature engineering utilities
- Sequence data handling capabilities
- Visualization and plotting tools
- Command-line interface for model training and evaluation

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2024-12-19

### Added
- Initial release of MLEX (Money Laundering Expert System)
- Core neural network models (GRU, LSTM, RNN) for sequence data
- Evaluation framework with standard metrics (AUC, F1, Precision, Recall)
- Data preprocessing pipeline with categorical and numerical transformers
- Feature stratified splitting for time series data
- Model evaluation and visualization tools
- Command-line interface for easy model training and evaluation
- Comprehensive documentation and examples

### Features
- **Models**: GRU, LSTM, RNN with configurable hyperparameters
- **Evaluation**: StandardEvaluator with multiple threshold strategies
- **Data Processing**: DataReader, FeatureStratifiedSplit, PreProcessingTransformer
- **Visualization**: EvaluationPlotter with ROC and PR curves
- **CLI**: Command-line interface for training and evaluation
- **Documentation**: Comprehensive README and contributing guidelines 