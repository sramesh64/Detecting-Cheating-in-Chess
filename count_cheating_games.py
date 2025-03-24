import pandas as pd

# Load the games
print("Loading game data...")
games = pd.read_csv('Games.csv')
print(f"Total games in dataset: {len(games)}")

# Count games where only one side is cheating
only_white_cheating = 0
only_black_cheating = 0
both_cheating = 0
none_cheating = 0

for idx, row in games.iterrows():
    # Convert string representation to list of integers
    white_cheat_list = [int(x) for x in row['Liste cheat white']]
    black_cheat_list = [int(x) for x in row['Liste cheat black']]
    
    white_cheating = sum(white_cheat_list) > 0
    black_cheating = sum(black_cheat_list) > 0
    
    if white_cheating and black_cheating:
        both_cheating += 1
    elif white_cheating and not black_cheating:
        only_white_cheating += 1
    elif not white_cheating and black_cheating:
        only_black_cheating += 1
    else:
        none_cheating += 1

print(f"Games with only WHITE cheating: {only_white_cheating}")
print(f"Games with only BLACK cheating: {only_black_cheating}")
print(f"Games with BOTH sides cheating: {both_cheating}")
print(f"Games with NO sides cheating: {none_cheating}")
print(f"Total games with only ONE side cheating: {only_white_cheating + only_black_cheating}")

# Verify that the counts sum up to the total
total = only_white_cheating + only_black_cheating + both_cheating + none_cheating
print(f"Sum of all categories: {total} (should equal total games: {len(games)})") 