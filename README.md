# ConSentiment
ConSentiment is a Python library for Aspect-Based Sentiment Analysis (ABSA) that leverages consensus among multiple models using graph learning. The library aims to reduce uncertainty in sentiment analysis by integrating various sentiment predictions through a graph-based approach.

## Features
- Consensus Information Graph: Connects similar reviews, aspects, and predicted labels to build a consensus.
- Uncertainty Reduction: Manages ambiguity by linking reviews with similar aspects but different sentiments.
- Graph-Based Regularization: Prioritizes consensual information to improve sentiment predictions.

## Installation
To install ConSentiment, you can use pip.
`pip install consentiment`

## Usage
Check the Committee-Graph Python Notebook in the examples folder.

## Methods Reference
`consentiment.create_graph_from_df(df: pd.DataFrame, text: str = 'text', ignore_model: List[str] = None, embedding_mi: float = 0.9, distance_threshold: float = 0.1, verbose: bool = True)`
Creates a consensus graph from a DataFrame.
- df: DataFrame containing the sentiment predictions.
- text: Column name for the review text.
- ignore_model: List of models to ignore, if a string is present in a df's column, that column will be skipped.
- embedding_mi: Weight for mixing aspect (mi) and text embeddings (1.0 - mi).
- distance_threshold: Threshold for connecting nodes in the graph.
- verbose: If True, prints progress information.

`consentiment.eval.evaluate(spans: pd.DataFrame, true_column: str = "annotation")`
Evaluates sentiment predictions through Nervaluate.
- spans: DataFrame with span annotations.
- true_column: Column name for true annotations.

`consentiment.eval.metrics_to_df(metrics: dict, eval_type: str = 'strict')`
Converts evaluation metrics to a DataFrame.
- metrics: Dictionary containing evaluation metrics.
- eval_type: Type of evaluation ('strict' or 'partial').