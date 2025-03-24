# train_cheating_detector.py - Training script for chess cheating detection model

import os
import io
import pandas as pd
import numpy as np
import chess
import chess.pgn
import chess.engine
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import time
import argparse
import random

def extract_game_features(pgn_str, stockfish_path=None, depth=10):
    """
    Extract legitimate game-level features for both players in a chess game.
    Focuses on chess quality metrics without using any cheating information.
    
    Args:
        pgn_str (str): The PGN string of the chess game
        stockfish_path (str): Path to Stockfish engine
        depth (int): Analysis depth for Stockfish
        
    Returns:
        dict: Features for both white and black players
    """
    # Initialize features for both sides with null values
    white_features = {
        'total_moves': 0,
        'avg_centipawn_loss': 0,
        'max_centipawn_loss': 0,
        'pct_best_moves': 0,
        'pct_top3_moves': 0,
        'position_eval_consistency': 0,  # Standard deviation of position evaluations
        'blunder_count': 0,
        'avg_eval_diff': 0,
        'early_queen': 0,            # Early queen development (opening principles)
        'knight_outpost_usage': 0,   # Knight positioning 
        'bishop_pair_retention': 0,  # Keeping bishop pair advantage
        'pawn_structure_score': 0,   # Pawn structure quality
        'king_safety': 0,            # King safety measure
        'piece_coordination': 0,     # Piece coordination score
        'endgame_technique': 0,      # Endgame technique quality
    }
    
    black_features = dict(white_features)  # Copy the structure
    
    # Skip engine analysis if no Stockfish path provided
    if not stockfish_path or not os.path.exists(stockfish_path):
        print("Stockfish not available, using dummy features")
        return {'white': white_features, 'black': black_features}
    
    try:
        # Set up the chess game
        pgn = io.StringIO(pgn_str)
        game = chess.pgn.read_game(pgn)
        if game is None:
            print("Failed to parse game")
            return {'white': white_features, 'black': black_features}
        
        # Initialize Stockfish engine
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 2})
        
        # Initialize tracking variables
        board = game.board()
        white_moves = 0
        black_moves = 0
        white_total_cp_loss = 0
        black_total_cp_loss = 0
        white_max_cp_loss = 0
        black_max_cp_loss = 0
        white_best_moves = 0
        black_best_moves = 0
        white_top3_moves = 0
        black_top3_moves = 0
        white_blunders = 0
        black_blunders = 0
        white_eval_diff = 0
        black_eval_diff = 0
        white_queen_move_ply = 100  # Default high value if queen isn't moved early
        black_queen_move_ply = 100
        
        # Track pawn structures
        white_pawn_structure = []
        black_pawn_structure = []
        
        # Track piece development
        white_piece_development = []
        black_piece_development = []
        
        # Keep previous evaluations to calculate consistency
        prev_evaluations = []
        
        # Process each move in the game
        for move_num, node in enumerate(game.mainline(), start=1):
            # Get the actual move played
            actual_move = node.move
            
            # Check for early queen moves
            if board.piece_at(actual_move.from_square) and board.piece_at(actual_move.from_square).piece_type == chess.QUEEN:
                if board.piece_at(actual_move.from_square).color == chess.WHITE and white_queen_move_ply == 100:
                    white_queen_move_ply = move_num
                elif board.piece_at(actual_move.from_square).color == chess.BLACK and black_queen_move_ply == 100:
                    black_queen_move_ply = move_num
            
            # Skip detailed analysis for opening moves
            if move_num <= 10:
                board.push(actual_move)
                continue
                
            # Analyze position before move
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
            best_moves = [entry["pv"][0] for entry in info]
            best_score = info[0]["score"].white()
            
            # Store evaluation for consistency tracking
            if best_score.is_mate():
                if best_score.mate() > 0:
                    eval_cp = 10000 - best_score.mate() * 10  # Winning mate
                else:
                    eval_cp = -10000 - best_score.mate() * 10  # Losing to mate
            else:
                eval_cp = best_score.score()
            
            prev_evaluations.append(eval_cp)
            
            # Convert score to centipawns
            if best_score.is_mate():
                if best_score.mate() > 0:
                    best_cp = 10000  # Winning mate
                else:
                    best_cp = -10000  # Losing to mate
            else:
                best_cp = best_score.score()
            
            # Make the move on the board
            board.push(actual_move)
            
            # Analyze position after move
            after_info = engine.analyse(board, chess.engine.Limit(depth=depth))
            after_score = after_info["score"].white()
            
            # Convert after-move score to centipawns
            if after_score.is_mate():
                if after_score.mate() > 0:
                    after_cp = 10000
                else:
                    after_cp = -10000
            else:
                after_cp = after_score.score()
            
            # Calculate centipawn loss
            is_white_move = not board.turn  # The move just made
            
            if is_white_move:
                # For white's move, calculate loss from white's perspective
                cp_loss = best_cp - after_cp
            else:
                # For black's move, calculate loss from black's perspective
                cp_loss = -(after_cp - best_cp)
            
            cp_loss = max(0, cp_loss)  # Only consider losses
            
            # Track statistics by player
            if is_white_move:
                white_moves += 1
                white_total_cp_loss += cp_loss
                white_max_cp_loss = max(white_max_cp_loss, cp_loss)
                if cp_loss > 100:
                    white_blunders += 1
                if actual_move == best_moves[0]:
                    white_best_moves += 1
                if actual_move in best_moves:
                    white_top3_moves += 1
                white_eval_diff += abs(best_cp - after_cp)
            else:
                black_moves += 1
                black_total_cp_loss += cp_loss
                black_max_cp_loss = max(black_max_cp_loss, cp_loss)
                if cp_loss > 100:
                    black_blunders += 1
                if actual_move == best_moves[0]:
                    black_best_moves += 1
                if actual_move in best_moves:
                    black_top3_moves += 1
                black_eval_diff += abs(best_cp - after_cp)
            
            # Track pawn structure (as a simple count of pawns by file)
            white_pawns = [square for square in chess.SQUARES if board.piece_at(square) and 
                          board.piece_at(square).piece_type == chess.PAWN and
                          board.piece_at(square).color == chess.WHITE]
            
            black_pawns = [square for square in chess.SQUARES if board.piece_at(square) and 
                          board.piece_at(square).piece_type == chess.PAWN and
                          board.piece_at(square).color == chess.BLACK]
                
            white_pawn_structure.append(len(white_pawns))
            black_pawn_structure.append(len(black_pawns))
            
            # Track piece development (pieces moved from starting squares)
            if move_num <= 20:  # Only track early development
                white_developed = sum(1 for square in range(8, 16) if not board.piece_at(square) or board.piece_at(square).color != chess.WHITE)
                black_developed = sum(1 for square in range(48, 56) if not board.piece_at(square) or board.piece_at(square).color != chess.BLACK)
                
                white_piece_development.append(white_developed)
                black_piece_development.append(black_developed)
        
        # Clean up engine
        engine.quit()
        
        # Calculate evaluation consistency (standard deviation)
        white_eval_consistency = np.std(prev_evaluations[::2]) if len(prev_evaluations) >= 2 else 0
        black_eval_consistency = np.std(prev_evaluations[1::2]) if len(prev_evaluations) >= 2 else 0
        
        # Calculate pawn structure consistency
        white_pawn_consistency = np.std(white_pawn_structure) if white_pawn_structure else 0
        black_pawn_consistency = np.std(black_pawn_structure) if black_pawn_structure else 0
        
        # Calculate development speed (average over first 10 moves)
        white_dev_speed = np.mean(white_piece_development[:10]) if white_piece_development else 0
        black_dev_speed = np.mean(black_piece_development[:10]) if black_piece_development else 0
        
        # Calculate final features
        if white_moves > 0:
            white_features['total_moves'] = white_moves
            white_features['avg_centipawn_loss'] = white_total_cp_loss / white_moves
            white_features['max_centipawn_loss'] = white_max_cp_loss
            white_features['pct_best_moves'] = white_best_moves / white_moves
            white_features['pct_top3_moves'] = white_top3_moves / white_moves
            white_features['position_eval_consistency'] = white_eval_consistency
            white_features['blunder_count'] = white_blunders
            white_features['avg_eval_diff'] = white_eval_diff / white_moves
            white_features['early_queen'] = 1.0 / max(1, white_queen_move_ply / 10)
            white_features['pawn_structure_score'] = white_pawn_consistency
            white_features['piece_coordination'] = white_dev_speed
            
        if black_moves > 0:
            black_features['total_moves'] = black_moves
            black_features['avg_centipawn_loss'] = black_total_cp_loss / black_moves
            black_features['max_centipawn_loss'] = black_max_cp_loss
            black_features['pct_best_moves'] = black_best_moves / black_moves
            black_features['pct_top3_moves'] = black_top3_moves / black_moves
            black_features['position_eval_consistency'] = black_eval_consistency
            black_features['blunder_count'] = black_blunders
            black_features['avg_eval_diff'] = black_eval_diff / black_moves
            black_features['early_queen'] = 1.0 / max(1, black_queen_move_ply / 10)
            black_features['pawn_structure_score'] = black_pawn_consistency
            black_features['piece_coordination'] = black_dev_speed
            
    except Exception as e:
        print(f"Error analyzing game: {e}")
    
    return {'white': white_features, 'black': black_features}

def create_feature_vector(features_dict):
    """
    Convert the features dictionary to a flat feature vector suitable for ML models.
    Also creates differential features comparing white vs black.
    FIXED to ensure consistent feature vector length across all games.
    """
    feature_vector = []
    feature_names = []
    
    # Add white features
    for name, value in features_dict['white'].items():
        feature_vector.append(value)
        feature_names.append(f'white_{name}')
    
    # Add black features
    for name, value in features_dict['black'].items():
        feature_vector.append(value)
        feature_names.append(f'black_{name}')
    
    # Add differential features (white - black)
    for name in features_dict['white'].keys():
        white_val = features_dict['white'][name]
        black_val = features_dict['black'][name]
        feature_vector.append(white_val - black_val)
        feature_names.append(f'diff_{name}')
    
    # Add ratio features with safe handling
    # Best move ratio (with safe division)
    w_best = features_dict['white']['pct_best_moves']
    b_best = features_dict['black']['pct_best_moves']
    ratio_best = w_best / max(0.001, b_best) if b_best != 0 else 1000.0
    feature_vector.append(ratio_best)
    feature_names.append('ratio_best_moves')
    
    # Centipawn loss ratio (with safe division)
    w_loss = features_dict['white']['avg_centipawn_loss']
    b_loss = features_dict['black']['avg_centipawn_loss']
    ratio_loss = b_loss / max(0.001, w_loss) if w_loss != 0 else 1000.0
    feature_vector.append(ratio_loss)
    feature_names.append('ratio_centipawn_loss')
        
    # Blunder ratio (with safe division)
    w_blunders = features_dict['white']['blunder_count'] + 0.1
    b_blunders = features_dict['black']['blunder_count'] + 0.1
    ratio_blunders = w_blunders / b_blunders
    feature_vector.append(ratio_blunders)
    feature_names.append('ratio_blunders')
    
    return feature_vector, feature_names

def find_one_side_cheating_games(games_df):
    """
    Find games where only one side is cheating.
    
    Args:
        games_df (DataFrame): DataFrame containing the games
        
    Returns:
        DataFrame: Filtered DataFrame with only one side cheating
    """
    one_side_cheating = []
    white_cheating_count = 0
    black_cheating_count = 0
    
    for idx, row in games_df.iterrows():
        # Convert string representation to list of integers
        white_cheat_list = [int(x) for x in row['Liste cheat white']]
        black_cheat_list = [int(x) for x in row['Liste cheat black']]
        
        white_cheating = sum(white_cheat_list) > 0
        black_cheating = sum(black_cheat_list) > 0
        
        if white_cheating and not black_cheating:
            one_side_cheating.append(idx)
            white_cheating_count += 1
        elif not white_cheating and black_cheating:
            one_side_cheating.append(idx)
            black_cheating_count += 1
    
    print(f"Found {white_cheating_count} games with only WHITE cheating")
    print(f"Found {black_cheating_count} games with only BLACK cheating")
    
    return games_df.loc[one_side_cheating]

def build_dataset(games_df, stockfish_path=None, max_games=None, balance_classes=True):
    """
    Build a dataset for training a legitimate model.
    IMPORTANT: Does not use the cheating information as features.
    
    Args:
        games_df (DataFrame): DataFrame containing games with one side cheating
        stockfish_path (str): Path to Stockfish engine
        max_games (int): Maximum number of games to process
        balance_classes (bool): If True, ensure balanced classes
        
    Returns:
        tuple: X, y, feature_names
    """
    X = []
    y = []
    feature_names = None
    
    # Count cheating by side
    white_cheating_indices = []
    black_cheating_indices = []
    
    for idx, row in games_df.iterrows():
        # Convert string representation to list of integers
        white_cheat_list = [int(x) for x in row['Liste cheat white']]
        black_cheat_list = [int(x) for x in row['Liste cheat black']]
        
        # Extract the cheating side (ground truth)
        white_cheating = sum(white_cheat_list) > 0
        black_cheating = sum(black_cheat_list) > 0
        
        # Skip games where both or neither side is cheating
        if (white_cheating and black_cheating) or (not white_cheating and not black_cheating):
            continue
        
        if white_cheating:
            white_cheating_indices.append(idx)
        else:
            black_cheating_indices.append(idx)
    
    print(f"Total white cheating games: {len(white_cheating_indices)}")
    print(f"Total black cheating games: {len(black_cheating_indices)}")
    
    # Balance classes if requested
    if balance_classes:
        min_size = min(len(white_cheating_indices), len(black_cheating_indices))
        if max_games and max_games < 2 * min_size:
            min_size = max_games // 2
        
        # Use fixed random seed for reproducibility
        np.random.seed(42)
        
        if len(white_cheating_indices) > min_size:
            white_cheating_indices = np.random.choice(white_cheating_indices, min_size, replace=False)
        
        if len(black_cheating_indices) > min_size:
            black_cheating_indices = np.random.choice(black_cheating_indices, min_size, replace=False)
        
        # Combine indices
        selected_indices = np.concatenate([white_cheating_indices, black_cheating_indices])
        subset_df = games_df.loc[selected_indices]
        
        print(f"Balanced dataset to {len(white_cheating_indices)} games per class")
    else:
        # Just limit total games if requested
        if max_games and max_games < len(games_df):
            subset_df = games_df.sample(max_games, random_state=42)
        else:
            subset_df = games_df
    
    print(f"Building dataset from {len(subset_df)} games...")
    
    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        # Convert string representation to list of integers
        white_cheat_list = [int(x) for x in row['Liste cheat white']]
        black_cheat_list = [int(x) for x in row['Liste cheat black']]
        
        # Extract the cheating side (ground truth)
        white_cheating = sum(white_cheat_list) > 0
        black_cheating = sum(black_cheat_list) > 0
        
        # Skip games where both or neither side is cheating (should already be filtered)
        if (white_cheating and black_cheating) or (not white_cheating and not black_cheating):
            continue
        
        # The label is 1 if white is cheating, 0 if black is cheating
        label = 1 if white_cheating else 0
        
        # Create PGN string if not already in DataFrame
        if 'pgn' not in row:
            pgn_str = f"""
[Event "?"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "Player1"]
[Black "Player2"]
[Result "*"]

{row['Game']} *
"""
        else:
            pgn_str = row['pgn']
        
        try:
            # Extract genuine chess features WITHOUT using cheating information
            features_dict = extract_game_features(pgn_str, stockfish_path)
            
            # Convert to feature vector
            feature_vector, current_feature_names = create_feature_vector(features_dict)
            
            # Store feature names from first game
            if feature_names is None:
                feature_names = current_feature_names
            
            # Verify that the feature vector has the expected length
            if len(feature_vector) == len(feature_names):
                X.append(feature_vector)
                y.append(label)
            else:
                print(f"Warning: Skipping game with inconsistent feature vector length: {len(feature_vector)} vs {len(feature_names)}")
        except Exception as e:
            print(f"Error processing game: {e}")
    
    if not X:
        raise ValueError("No valid feature vectors were extracted. Check your game data and Stockfish path.")
    
    # Convert to numpy arrays, ensuring consistent shapes
    X_array = np.array(X)
    y_array = np.array(y)
    
    print(f"Final dataset shape: {X_array.shape}")
    
    return X_array, y_array, feature_names

def train_and_evaluate_model(X, y, feature_names, output_dir='results/training'):
    """
    Train and evaluate multiple models for cheating side detection.
    
    Args:
        X (array): Feature matrix
        y (array): Target vector (1 for white cheating, 0 for black cheating)
        feature_names (list): List of feature names
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Dictionary with best model and metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Print class distribution
    print("Training set class distribution:")
    train_dist = pd.Series(y_train).value_counts(normalize=True)
    print(train_dist)
    
    # List of models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ])
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    
    # For storing ROC curves
    plt.figure(figsize=(10, 8))
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_score = -1
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores[name] = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"Cross-validation ROC-AUC: {cv_scores[name].mean():.3f} ± {cv_scores[name].std():.3f}")
        
        # Train on the full training set
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_scores': cv_scores[name]
        }
        
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {name}')
        plt.colorbar()
        plt.xticks([0, 1], ['BLACK', 'WHITE'])
        plt.yticks([0, 1], ['BLACK', 'WHITE'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name.lower().replace(" ", "_")}.png'))
        plt.close()
        
        # Track the best model
        if results[name]['f1'] > best_score:
            best_score = results[name]['f1']
            best_model = model
            best_name = name
    
    # Plot cross-validation results
    plt.figure(figsize=(12, 6))
    boxplot_data = [cv_scores[name] for name in models.keys()]
    plt.boxplot(boxplot_data, labels=list(models.keys()))
    plt.title('Cross-Validation ROC-AUC Scores')
    plt.ylabel('ROC-AUC')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_results.png'))
    plt.close()
    
    # Visualize feature importances for the best model if it's a random forest
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
        plt.close()
    elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        importances = best_model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
        plt.close()
    
    # Prepare the model data to save
    model_data = {
        'model': best_model,
        'feature_names': feature_names,
        'metadata': {
            'accuracy': results[best_name]['accuracy'],
            'precision': results[best_name]['precision'],
            'recall': results[best_name]['recall'],
            'f1': results[best_name]['f1'],
            'cv_scores': results[best_name]['cv_scores'].tolist(),
            'model_type': best_name
        },
        'metrics': results[best_name]
    }
    
    # Save the model to a file
    with open(os.path.join('models', 'cheating_detector.pkl'), 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nBest model ({best_name}) saved to models/cheating_detector.pkl")
    
    # Save the model metrics to a JSON file
    import json
    metrics_file = os.path.join(output_dir, 'model_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({k: v for k, v in model_data['metadata'].items() if k != 'cv_scores'}, f, indent=4)
    print(f"Model metrics saved to {metrics_file}")
    
    return model_data

def main():
    parser = argparse.ArgumentParser(description='Train a model to predict which side is cheating in chess games')
    parser.add_argument('--input', type=str, default='Games.csv', help='Path to the input CSV file')
    parser.add_argument('--max_games', type=int, help='Maximum number of games to process')
    parser.add_argument('--stockfish_path', type=str, help='Path to Stockfish engine')
    parser.add_argument('--output_dir', type=str, default='results/training', help='Directory to save output files')
    args = parser.parse_args()
    
    # Load the games
    print(f"Loading games from {args.input}...")
    games_df = pd.read_csv(args.input)
    
    # Build the dataset
    X, y, feature_names = build_dataset(games_df, stockfish_path=args.stockfish_path, max_games=args.max_games)
    
    # Train and evaluate the model
    model_data = train_and_evaluate_model(X, y, feature_names, output_dir=args.output_dir)
    
    print("Model training and evaluation complete!")

if __name__ == "__main__":
    main() 