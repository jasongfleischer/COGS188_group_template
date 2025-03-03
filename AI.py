from tetris import Tetris, Figure
from typing import List, Tuple

def get_legal_placements(tetris: Tetris, piece: Figure) -> List[Tuple[int, int]]:
    """
    Check design doc for details
    """
    # TODO: Implement
    pass

def simulate_placement(tetris: Tetris, piece: Figure, rotation: int, x: int) -> Tetris:
    """
    Check design doc for details
    """
    # TODO: Implement
    pass

def evaluate_state(tetris: Tetris) -> float:
    """
    Check design doc for details
    """
    # TODO: Implement
    pass

def backtracking_search(tetris: Tetris, pieces: List[Figure], depth: int) -> Tuple[List[Tuple[int, int]], float]:
    """
    TODO: Pierce - fill this out
    """
    #if depth is 0 or no pieces are left, return the current board status
    if depth == 0 or not pieces:
        return [], evaluate_state(tetris)
    
    best_sequence: List[Tuple[int, int]] = []
    best_score = float('-inf')
    
    #start at the first piece
    current_piece = pieces[0]
    legal_moves = get_legal_placements(tetris, current_piece)
    
    #for every legal move recursively take them
    for move in legal_moves:
        rotation, x = move
        #simulate the move then recurse
        new_state = simulate_placement(tetris, current_piece, rotation, x)
        subsequent_moves, score = backtracking_search(new_state, pieces[1:], depth - 1)
        #update best score and move accordingly
        if score > best_score:
            best_score = score
            best_sequence = [move] + subsequent_moves
    
    return best_sequence, best_score

def get_next_moves(tetris: Tetris, pieces: List[Figure]) -> List[Tuple[int, int]]:
    """
    TODO: Pierce - fill this out
    """
    #generate our next 10 moves using backtracking
    depth = min(10, len(pieces))
    moves, _ = backtracking_search(tetris, pieces, depth)
    #this will return to tetris.py
    return moves