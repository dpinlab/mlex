from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlex.evaluation.evaluator import StandardEvaluator
from mlex.evaluation.threshold import F1MaxThresholdStrategy

# Creating dataset
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Building pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)

scores = pipeline.predict_proba(X_test)[:, 1] # Getting predictions and scores

evaluator = StandardEvaluator("test_model_3", F1MaxThresholdStrategy())
evaluator.evaluate(y_test, [], scores) # Evaluate
print(evaluator.summary())

evaluator.save("evaluation.parquet") # Saving results

print("\nFile summary:")
print(StandardEvaluator.parquet_summary("evaluation.parquet")) # Loading and summarize
