# Chess Cheating Detection Project

This project uses machine learning to detect cheating in chess games. The model analyzes patterns in move quality and strategic choices to identify when players are using chess engines like Stockfish to gain an unfair advantage. When cheating is detected, the model can determine whether it's the white or black player who is using computer assistance.

## Overview

The system uses a dataset of 48,000+ games where each move was made by either:
- Maia (a neural network that plays like a human of a specific rating)
- Stockfish (a top chess engine)

By training on this labeled data, our model learns to distinguish human-like moves from computer-generated ones.

## Features

- Extract features from chess games including:
  - Move quality metrics:
    - Centipawn loss (difference between played move and Stockfish's best move)
    - Best move percentage (how often player chooses Stockfish's top recommendation)
    - Top 3 move percentage (how often player chooses one of Stockfish's top 3 moves)
    - Blunder frequency and magnitude
    - Position evaluation consistency over time
  - Chess principles and strategy:
    - Early queen development detection
    - Piece development speed (from starting positions)
    - Pawn structure consistency
  - Comparative metrics:
    - Difference in metrics between white and black
    - Ratios of key metrics (best moves, centipawn loss, blunders)
  
- Train multiple machine learning models:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression

## Interpreting Results

The evaluation results show how well the model can detect which side in a chess game is using computer assistance:

- **Accuracy**: Overall proportion of correct predictions
- **Precision**: Proportion of positive predictions that were correct
- **Recall**: Proportion of actual positives that were correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve, indicating model discriminative power

## Limitations

- The model is trained on specific types of chess engines (Stockfish and Maia)
- Strong human players may trigger false positives by naturally playing engine-like moves
- The system works best on complete games rather than short fragments
- Analysis requires Stockfish to be installed and accessible


## Acknowledgments

Chess Cheating Dataset: https://www.kaggle.com/datasets/brieucdandoy/chess-cheating-dataset
