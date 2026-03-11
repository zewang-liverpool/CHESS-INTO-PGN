import chess.pgn

# Define the target PGN artifact path for validation
# (Update this to "extracted_game_static.pgn" if validating your newest pipeline output)
pgn_file_path = "final_game.pgn"

try:
    with open(pgn_file_path, "r", encoding="utf-8") as f:
        # Parse the PGN file into a python-chess game object
        game = chess.pgn.read_game(f)
        
    if game is None:
        print("[Error] PGN file is empty or heavily corrupted.")
    else:
        board = game.board()
        print("[Init] Initial board state:")
        print(board)
        print("-" * 20)
        
        # Iterate through the main line moves to sequentially reconstruct the game state
        for index, move in enumerate(game.mainline_moves()):
            board.push(move)
            print(f"\n[Move {index + 1}] Executing: {move}")
            # Render the current board state in ASCII format
            print(board) 
            
        print("\n[Success] Game reconstruction complete.")
        print(f"[Result] Final FEN string: {board.fen()}")

except FileNotFoundError:
    print(f"[Error] Target file not found: {pgn_file_path}")