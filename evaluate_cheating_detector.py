# evaluate_cheating_detector.py - Evaluation script for chess cheating detection model

import os
import io
import pandas as pd
import numpy as np
import chess
import chess.pgn
import chess.engine
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

def load_model(model_path):
    """
    Load the trained model and print its metrics.
    
    Args:
        model_path (str): Path to the trained model
        
    Returns:
        dict: Dictionary containing the model, feature names, and metrics
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Print model metrics
        metrics = model_data.get('metrics', {})
        print(f"Loaded model from {model_path}")
        print(f"Model cross-validation accuracy: {metrics.get('cv_scores', [0])[0]:.4f}")
        print(f"Model test accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"Model precision: {metrics.get('precision', 0):.4f}")
        print(f"Model recall: {metrics.get('recall', 0):.4f}")
        print(f"Model F1 score: {metrics.get('f1', 0):.4f}")
        
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_model(model_data, games_df, stockfish_path=None, max_games=None, output_dir='results/evaluation'):
    """
    Evaluate the model on a dataset of games.
    
    Args:
        model_data (dict): Dictionary containing the model and metadata
        games_df (DataFrame): DataFrame containing the games
        stockfish_path (str): Path to the Stockfish engine
        max_games (int): Maximum number of games to evaluate
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find games where only one side is cheating
    one_side_games = find_one_side_cheating_games(games_df)
    print(f"Found {len(one_side_games)} games with only one side cheating")
    
    # Limit the number of games if specified
    if max_games and max_games < len(one_side_games):
        one_side_games = one_side_games.sample(max_games, random_state=42)
        print(f"Evaluating on {len(one_side_games)} random games")
    
    # Get the model
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Initialize lists for results
    y_true = []
    y_pred = []
    confidences = []
    
    # Process each game
    print("Processing games...")
    for idx, row in tqdm(one_side_games.iterrows(), total=len(one_side_games)):
        # Create PGN
        pgn_str = f"""[Event "?"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "*"]

{row['Game']} *"""
        
        # Extract features
        features_dict = extract_game_features(pgn_str, stockfish_path)
        feature_vector, _ = create_feature_vector(features_dict)
        
        # Match features with model's expected features
        if len(feature_vector) != len(feature_names):
            print(f"Warning: Feature vector length mismatch for game {idx} (expected {len(feature_names)}, got {len(feature_vector)})")
            
            # Try to create a feature vector with the right size
            feature_vector = create_consistent_feature_vector(features_dict, feature_names)
            
            if len(feature_vector) != len(feature_names):
                print(f"Skipping game {idx} due to feature mismatch")
                continue
        
        # Make prediction
        pred = model.predict([feature_vector])[0]
        prob = model.predict_proba([feature_vector])[0]
        confidence = prob.max()
        
        # Determine true label (whether white or black is cheating)
        white_cheating = sum(int(x) for x in row['Liste cheat white']) > 0
        black_cheating = sum(int(x) for x in row['Liste cheat black']) > 0
        
        if white_cheating and not black_cheating:
            true_label = 1  # White is cheating
        elif black_cheating and not white_cheating:
            true_label = 0  # Black is cheating
        else:
            print(f"Warning: Game {idx} has neither or both sides cheating")
            continue
        
        # Store results
        y_true.append(true_label)
        y_pred.append(pred)
        confidences.append(confidence)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average confidence: {np.mean(confidences):.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print matrix
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                BLACK  WHITE")
    print(f"True BLACK:      {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"True WHITE:      {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Calculate percentage of correct predictions by true side
    white_true = np.array(y_true) == 1
    black_true = np.array(y_true) == 0
    white_correct = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    black_correct = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
    
    white_accuracy = white_correct/np.sum(white_true) if np.sum(white_true) > 0 else 0
    black_accuracy = black_correct/np.sum(black_true) if np.sum(black_true) > 0 else 0
    
    print(f"\nAccuracy on WHITE cheating games: {white_accuracy:.4f} ({white_correct}/{np.sum(white_true)})")
    print(f"Accuracy on BLACK cheating games: {black_accuracy:.4f} ({black_correct}/{np.sum(black_true)})")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    plt.close()
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    correct = np.array(y_true) == np.array(y_pred)
    
    plt.hist([np.array(confidences)[correct], np.array(confidences)[~correct]], 
             bins=10, label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
    plt.xlabel('Confidence')
    plt.ylabel('Number of Predictions')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    print(f"Confidence distribution saved to {os.path.join(output_dir, 'confidence_distribution.png')}")
    plt.close()
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    
    # Get the predicted probabilities for the positive class
    y_probs = []
    for idx, row in tqdm(one_side_games.head(len(y_true)).iterrows(), total=len(y_true)):
        pgn_str = f"""[Event "?"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "*"]

{row['Game']} *"""
        
        features_dict = extract_game_features(pgn_str, stockfish_path)
        feature_vector, _ = create_feature_vector(features_dict)
        
        if len(feature_vector) != len(feature_names):
            feature_vector = create_consistent_feature_vector(features_dict, feature_names)
        
        prob = model.predict_proba([feature_vector])[0][1]  # Probability of white cheating
        y_probs.append(prob)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    print(f"ROC curve saved to {os.path.join(output_dir, 'roc_curve.png')}")
    plt.close()
    
    # Save metrics to a JSON file
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'average_confidence': float(np.mean(confidences)),
        'white_accuracy': white_accuracy,
        'black_accuracy': black_accuracy,
        'roc_auc': roc_auc,
        'num_games': len(y_true)
    }
    
    import json
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_file}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'confidences': confidences,
        'y_true': y_true,
        'y_pred': y_pred,
        'roc_auc': roc_auc
    }

def create_consistent_feature_vector(features_dict, expected_feature_names):
    """
    Create a feature vector that matches the expected feature names.
    This is used when the standard feature vector has a different length than what the model expects.
    
    Args:
        features_dict (dict): Dictionary of features extracted from a game
        expected_feature_names (list): List of feature names expected by the model
        
    Returns:
        list: Feature vector with values corresponding to expected_feature_names
    """
    feature_vector = []
    
    # Define mappings from feature names to values in the features_dict
    feature_mapping = {
        # White player features
        'white_avg_centipawn_loss': features_dict['white'].get('avg_centipawn_loss', 0),
        'white_max_centipawn_loss': features_dict['white'].get('max_centipawn_loss', 0),
        'white_avg_eval_diff': features_dict['white'].get('avg_eval_diff', 0),
        'white_pct_best_moves': features_dict['white'].get('pct_best_moves', 0),
        'white_pct_top3_moves': features_dict['white'].get('pct_top3_moves', 0),
        'white_total_moves': features_dict['white'].get('total_moves', 0),
        
        # Black player features
        'black_avg_centipawn_loss': features_dict['black'].get('avg_centipawn_loss', 0),
        'black_max_centipawn_loss': features_dict['black'].get('max_centipawn_loss', 0),
        'black_avg_eval_diff': features_dict['black'].get('avg_eval_diff', 0),
        'black_pct_best_moves': features_dict['black'].get('pct_best_moves', 0),
        'black_pct_top3_moves': features_dict['black'].get('pct_top3_moves', 0),
        'black_total_moves': features_dict['black'].get('total_moves', 0),
        
        # Difference features
        'diff_avg_centipawn_loss': features_dict.get('diff_avg_centipawn_loss', 0),
        'diff_max_centipawn_loss': features_dict.get('diff_max_centipawn_loss', 0),
        'diff_avg_eval_diff': features_dict.get('diff_avg_eval_diff', 0),
        'diff_pct_best_moves': features_dict.get('diff_pct_best_moves', 0),
        'diff_pct_top3_moves': features_dict.get('diff_pct_top3_moves', 0),
        'diff_total_moves': features_dict.get('diff_total_moves', 0),
        'diff_position_eval_consistency': features_dict.get('diff_position_eval_consistency', 0),
        'diff_pawn_structure_score': features_dict.get('diff_pawn_structure_score', 0),
        
        # Ratio features
        'ratio_best_moves': features_dict.get('ratio_best_moves', 1),
        'ratio_centipawn_loss': features_dict.get('ratio_centipawn_loss', 1),
        'ratio_blunders': features_dict.get('ratio_blunders', 1)
    }
    
    # Build feature vector using expected feature names
    for name in expected_feature_names:
        if name in feature_mapping:
            feature_vector.append(feature_mapping[name])
        else:
            # For unknown features, use 0 as a default value
            print(f"Warning: Unknown feature '{name}' in model, using default value 0")
            feature_vector.append(0)
    
    return feature_vector

def main():
    parser = argparse.ArgumentParser(description='Evaluate the performance of a chess cheating side detection model')
    parser.add_argument('--model', type=str, default='models/cheating_detector.pkl', help='Path to the model file')
    parser.add_argument('--input', type=str, default='Games.csv', help='Path to the input CSV file')
    parser.add_argument('--max_games', type=int, help='Maximum number of games to evaluate')
    parser.add_argument('--stockfish_path', type=str, help='Path to Stockfish engine')
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help='Directory to save output files')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model_data = load_model(args.model)
    
    # Load games
    print(f"Loading games from {args.input}...")
    games_df = pd.read_csv(args.input)
    
    # Evaluate model
    results = evaluate_model(model_data, games_df, args.stockfish_path, args.max_games, args.output_dir)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main() 