import pandas as pd
import chess
import chess.pgn
import io
import numpy as np
from stockfish import Stockfish
import os
import re
import argparse
import multiprocessing
from tqdm import tqdm
import functools

# Configure Stockfish - update the path to your Stockfish executable
stockfish_path = "/usr/local/bin/stockfish"  # Change this to your Stockfish path
if not os.path.exists(stockfish_path):
    stockfish_path = "stockfish"  # Try without path if file doesn't exist

# Create a function to initialize stockfish for each process
def initialize_stockfish():
    try:
        return Stockfish(path=stockfish_path, depth=10)
    except Exception as e:
        print(f"Error initializing Stockfish: {e}")
        return None

# Global stockfish instance for non-parallel code
try:
    stockfish = initialize_stockfish()
except Exception as e:
    print(f"Error initializing Stockfish: {e}")
    print("You may need to install Stockfish and update the path in the script.")
    stockfish = None

def extract_game_moves(pgn_str):
    """Parse PGN string and extract moves as a list."""
    # Extract just the moves, removing annotations, comments, etc.
    # We're handling the result part which appears at the end of the Game field
    # Remove the result at the end (1-0, 0-1, 1/2-1/2)
    pgn_str = re.sub(r'\s+[0-9/\-]+$', '', pgn_str)
    
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    if not game:
        return None
    
    # Extract moves
    moves = []
    board = game.board()
    for move in game.mainline_moves():
        moves.append((board.san(move), move))
        board.push(move)
    
    return moves, game

def calculate_centipawn_loss(board, move, stockfish):
    """Calculate centipawn loss for a move using Stockfish."""
    if stockfish is None:
        return None
    
    # Get position before move
    stockfish.set_fen_position(board.fen())
    best_move_info = stockfish.get_top_moves(1)
    if not best_move_info:
        return None
    
    # Handle mate score or missing centipawn value
    if 'Centipawn' not in best_move_info[0]:
        # This is likely a mate score, assign a large value
        if 'Mate' in best_move_info[0]:
            mate_score = best_move_info[0]['Mate']
            # Convert mate score to centipawn (mate in 1 is better than mate in 5)
            # By convention, we use ±10000 for mate
            best_eval = 10000 if mate_score > 0 else -10000
            # Adjust slightly based on number of moves to mate
            best_eval = best_eval - (mate_score if mate_score > 0 else -mate_score)
        else:
            # If we can't determine the evaluation, skip this move
            return None
    else:
        best_eval = best_move_info[0]['Centipawn']
    
    # Apply the actual move
    board.push(move)
    stockfish.set_fen_position(board.fen())
    
    # Get evaluation after the move
    position_info = stockfish.get_evaluation()
    if 'value' not in position_info:
        board.pop()
        return None
    
    actual_eval = position_info['value']
    
    # Check if this is a mate evaluation
    if 'type' in position_info and position_info['type'] == 'mate':
        mate_score = position_info['value']
        # Convert mate score to centipawn value
        actual_eval = 10000 if mate_score > 0 else -10000
        # Adjust slightly based on number of moves to mate
        actual_eval = actual_eval - (mate_score if mate_score > 0 else -mate_score)
    
    # Calculate centipawn loss
    # If it's black's move, negate the evaluations
    if not board.turn:  # If it's white's turn now, the move was by black
        best_eval = -best_eval if best_eval is not None else None
        actual_eval = -actual_eval if actual_eval is not None else None
    
    # Safety check to make sure we have valid evaluations
    if best_eval is None or actual_eval is None:
        board.pop()
        return None
    
    centipawn_loss = max(0, best_eval - actual_eval)
    
    # Undo the move
    board.pop()
    
    return centipawn_loss

def is_top_engine_move(board, move, stockfish, n=3):
    """Check if the move is one of Stockfish's top n choices."""
    if stockfish is None:
        return None
    
    try:
        stockfish.set_fen_position(board.fen())
        top_moves = stockfish.get_top_moves(n)
        if not top_moves:
            return None
            
        top_move_uci = [m['Move'] for m in top_moves]
        return move.uci() in top_move_uci
    except Exception as e:
        print(f"Error in is_top_engine_move: {e}")
        return None

def get_position_features(board):
    """Extract features from the current board position."""
    # Material count
    material = {
        'P': len(board.pieces(chess.PAWN, chess.WHITE)),
        'N': len(board.pieces(chess.KNIGHT, chess.WHITE)),
        'B': len(board.pieces(chess.BISHOP, chess.WHITE)),
        'R': len(board.pieces(chess.ROOK, chess.WHITE)),
        'Q': len(board.pieces(chess.QUEEN, chess.WHITE)),
        'p': len(board.pieces(chess.PAWN, chess.BLACK)),
        'n': len(board.pieces(chess.KNIGHT, chess.BLACK)),
        'b': len(board.pieces(chess.BISHOP, chess.BLACK)),
        'r': len(board.pieces(chess.ROOK, chess.BLACK)),
        'q': len(board.pieces(chess.QUEEN, chess.BLACK))
    }
    
    # Material balance (positive for white advantage)
    material_balance = (
        material['P'] - material['p'] + 
        3 * (material['N'] - material['n']) + 
        3 * (material['B'] - material['b']) + 
        5 * (material['R'] - material['r']) + 
        9 * (material['Q'] - material['q'])
    )
    
    # Game phase estimate
    total_pieces = sum(material.values())
    game_phase = 'opening' if total_pieces >= 30 else 'endgame' if total_pieces <= 12 else 'middlegame'
    
    # Control of center
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    white_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
    black_center_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
    
    return {
        'material_balance': material_balance,
        'game_phase': game_phase,
        'white_center_control': white_center_control,
        'black_center_control': black_center_control,
        'is_check': board.is_check(),
        'total_pieces': total_pieces
    }

def extract_move_features(game_row, engine=None, skip_expensive=False):
    """Extract features for each move in the game."""
    # Use the provided engine or fallback to global stockfish
    stockfish_engine = engine if engine is not None else stockfish
    
    try:
        pgn_str = game_row['Game']
        white_cheating_list = game_row['Liste cheat white']
        black_cheating_list = game_row['Liste cheat black']
        white_elo = game_row['Elo White']
        black_elo = game_row['Elo Black']
        
        moves_data = extract_game_moves(pgn_str)
        if not moves_data:
            print("Failed to extract game moves")
            return []
        
        moves, game = moves_data
        
        # For clarity, calculate the total number of full moves
        total_full_moves = (len(moves) + 1) // 2  # Round up to account for games ending on white's move
        
        # Check if the cheating lists are long enough for the game
        if total_full_moves > len(white_cheating_list) or total_full_moves > len(black_cheating_list):
            print(f"Warning: Cheating list too short for game with {total_full_moves} moves. " 
                  f"White list: {len(white_cheating_list)}, Black list: {len(black_cheating_list)}")
        
        move_features = []
        board = chess.Board()
        
        for move_idx, (san_move, move) in enumerate(moves):
            try:
                is_white_move = move_idx % 2 == 0
                player_elo = white_elo if is_white_move else black_elo
                
                # Calculate the full move number (1-indexed)
                full_move_number = (move_idx // 2) + 1
                
                # Get the correct cheating flag for this specific move
                if is_white_move:
                    # White moves correspond to the move number in the white list (0-indexed)
                    list_idx = full_move_number - 1  # Convert 1-indexed move to 0-indexed list
                    is_cheating = white_cheating_list[list_idx] if list_idx < len(white_cheating_list) else '0'
                else:
                    # Black moves also correspond to the move number in the black list
                    list_idx = full_move_number - 1  # Convert 1-indexed move to 0-indexed list
                    is_cheating = black_cheating_list[list_idx] if list_idx < len(black_cheating_list) else '0'
                
                move_type = 'stockfish' if is_cheating == '1' else 'maia'
                
                # Skip the first 10 moves as they're from the Lichess database
                if full_move_number <= 10:  # First 10 moves
                    board.push(move)
                    continue
                
                position_features = get_position_features(board)
                
                # Calculate move-specific features if not skipping expensive operations
                centipawn_loss = None
                is_top_move = None
                
                if not skip_expensive and stockfish_engine is not None:
                    centipawn_loss = calculate_centipawn_loss(board, move, stockfish_engine)
                    is_top_move = is_top_engine_move(board, move, stockfish_engine)
                
                move_data = {
                    'game_id': game_row.name,
                    'move_number': full_move_number,
                    'is_white': is_white_move,
                    'player_elo': player_elo,
                    'san_move': san_move,
                    'uci_move': move.uci(),
                    'centipawn_loss': centipawn_loss,
                    'is_top_engine_move': is_top_move,
                    'is_capture': board.is_capture(move),
                    'is_check': board.gives_check(move),
                    'is_cheating': is_cheating == '1',
                    'move_type': move_type
                }
                
                # Add position features
                move_data.update(position_features)
                
                move_features.append(move_data)
            except Exception as e:
                print(f"Error processing move {move_idx} ({san_move}): {e}")
                # Continue with the next move even if there was an error
            
            # Update the board with the move
            board.push(move)
        
        return move_features
    except Exception as e:
        print(f"Error in extract_move_features: {e}")
        return []

def process_game(game_row, skip_expensive=False):
    """Process a single game. Used for parallel processing."""
    # Initialize a stockfish instance for this process
    local_stockfish = initialize_stockfish()
    
    try:
        print(f"Processing game {game_row.name}")
        # Modified to use local stockfish instance
        move_features = extract_move_features(game_row, engine=local_stockfish, skip_expensive=skip_expensive)
        return move_features
    except Exception as e:
        print(f"Error processing game {game_row.name}: {e}")
        return []

def process_games(csv_file, max_games=100, parallel=True, num_processes=None, skip_expensive=False):
    """Process games from CSV file and extract move-level features."""
    # Load data
    try:
        df = pd.read_csv(csv_file)
        
        # Limit to max_games for initial testing
        if max_games:
            df = df.head(max_games)
        
        all_move_features = []
        
        if parallel and max_games > 1:
            # Use multiprocessing for faster processing
            if num_processes is None:
                # Use half the available cores by default
                num_processes = max(1, multiprocessing.cpu_count() // 2)
            
            print(f"Using {num_processes} processes for parallel processing")
            
            # Create a pool of workers
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Process games in parallel with partial to pass skip_expensive
                partial_process = functools.partial(process_game, skip_expensive=skip_expensive)
                results = list(tqdm(pool.imap(partial_process, [row for idx, row in df.iterrows()]), 
                                total=len(df), desc="Processing games"))
                
                # Combine all results
                for result in results:
                    all_move_features.extend(result)
        else:
            # Sequential processing
            for idx, row in df.iterrows():
                try:
                    print(f"Processing game {idx+1}/{len(df)}")
                    move_features = extract_move_features(row, skip_expensive=skip_expensive)
                    all_move_features.extend(move_features)
                    
                    # Save progress periodically
                    if idx > 0 and idx % 10 == 0:
                        temp_df = pd.DataFrame(all_move_features)
                        temp_df.to_csv(f'move_features_progress_{idx}.csv', index=False)
                except Exception as e:
                    print(f"Error processing game {idx+1}: {e}")
                    # Continue with the next game even if there was an error
        
        # Create final dataframe
        if all_move_features:
            moves_df = pd.DataFrame(all_move_features)
            moves_df.to_csv('move_features.csv', index=False)
            return moves_df
        else:
            print("No move features were extracted. Check for errors above.")
            return None
    except Exception as e:
        print(f"Error in process_games: {e}")
        return None

if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Process chess games and extract features')
    parser.add_argument('--test', action='store_true', help='Run in test mode with only 5 games')
    parser.add_argument('--games', type=int, default=100, help='Number of games to process')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--processes', type=int, help='Number of processes to use (default: half of CPU cores)')
    parser.add_argument('--fast', action='store_true', help='Skip expensive engine analysis for faster processing')
    args = parser.parse_args()
    
    # Process the games and extract move-level features
    if args.test:
        print("Running in TEST MODE with 5 games")
        process_games("Games.csv", max_games=5, parallel=not args.no_parallel, 
                     num_processes=args.processes, skip_expensive=args.fast)
    else:
        process_games("Games.csv", max_games=args.games, parallel=not args.no_parallel, 
                     num_processes=args.processes, skip_expensive=args.fast)
    
    print("Feature extraction complete. Results saved to move_features.csv") 