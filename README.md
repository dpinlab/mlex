# Money Laundering Expert System (MLEX)

A comprehensive machine learning framework for financial fraud detection and money laundering prevention.

## Features

- **Neural Network Models**: GRU, LSTM, and RNN implementations optimized for sequence data
- **Evaluation Framework**: Comprehensive evaluation metrics and visualization tools
- **Data Processing**: Advanced preprocessing and feature engineering capabilities
- **Model Pipeline**: End-to-end machine learning pipelines for fraud detection
- **Visualization**: Interactive plotting and analysis tools

## Installation

```bash
pip install mlex-lib
```

## Quick Start

```python
import pandas as pd
import numpy as np
from mlex.models import GRU, LSTM, RNN
from mlex.utils import DataReader, FeatureStratifiedSplit
from mlex.evaluation import StandardEvaluator, F1MaxThresholdStrategy

# Load and preprocess data
reader = DataReader('path/to/your/data.csv', target_columns=['fraud_label'])
X = reader.fit_transform()
y = reader.get_target()

# Split data
splitter = FeatureStratifiedSplit(column_to_stratify='account_id', test_proportion=0.3)
splitter.fit(X, y)
X_train, y_train, X_test, y_test = splitter.transform(X, y)

# Train model
model = GRU(
    target_column='fraud_label',
    validation_data=(X_test, y_test),
    input_size=10,
    hidden_size=64,
    epochs=50
)
model.fit(X_train, y_train)

# Evaluate
scores = model.score_samples(X_test)
evaluator = StandardEvaluator("fraud_detection", F1MaxThresholdStrategy())
evaluator.evaluate(y_test, [], scores)
print(evaluator.summary())
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MLEX in your research, please cite:

```bibtex
@software{mlex2024,
  title={Money Laundering Expert System (MLEX)},
  author={Pinheiro, Diego},
  year={2024},
  url={https://github.com/IoTDataAtelier/mlex}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/IoTDataAtelier/mlex/issues)
- **Discussions**: [GitHub Discussions](https://github.com/IoTDataAtelier/mlex/discussions)
