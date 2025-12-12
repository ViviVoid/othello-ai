# Andy Dao (daoa@msoe.edu)
# Othello Pygame

import pygame
import argparse
import json
import sys
import os
import numpy as np
import time
from datetime import datetime
from agent import HumanAgent, RandomMCTSAgent, MinimaxAgent, RandomAgent

# --- Game constants ---
BOARD_SIZE = 8
CELL_SIZE = 80
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE
GREEN = (0, 128, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
HIGHLIGHT = (200, 200, 0)

# Initialize pygame only if needed (for headless mode)
screen = None
font = None
pygame_initialized = False


def init_pygame():
    """Initialize pygame for display mode."""
    global screen, font, pygame_initialized
    if not pygame_initialized:
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
        pygame.display.set_caption("Othello (Reversi)")
        font = pygame.font.SysFont(None, 36)
        pygame_initialized = True


# --- Core Game Logic ---
def create_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    mid = BOARD_SIZE // 2
    board[mid - 1][mid - 1] = 1
    board[mid][mid] = 1
    board[mid - 1][mid] = -1
    board[mid][mid - 1] = -1
    return board


def in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def get_valid_moves(board, player):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    valid_moves = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != 0:
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                found_opponent = False
                while in_bounds(nr, nc) and board[nr][nc] == -player:
                    found_opponent = True
                    nr += dr
                    nc += dc
                if found_opponent and in_bounds(nr, nc) and board[nr][nc] == player:
                    valid_moves.append((r, c))
                    break
    return valid_moves


def apply_move(board, move, player):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    new_board = np.copy(board)
    r, c = move
    new_board[r][c] = player

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        to_flip = []
        while in_bounds(nr, nc) and new_board[nr][nc] == -player:
            to_flip.append((nr, nc))
            nr += dr
            nc += dc
        if in_bounds(nr, nc) and new_board[nr][nc] == player:
            for fr, fc in to_flip:
                new_board[fr][fc] = player
    return new_board


def count_discs(board):
    whites = int(np.sum(board == 1))
    blacks = int(np.sum(board == -1))
    return whites, blacks


# --- Board State Heuristics ---
def get_mobility(board, player):
    """Calculate mobility: number of valid moves available to the player."""
    return len(get_valid_moves(board, player))


def count_stable_discs(board, player):
    """
    Calculate stable discs for a player.
    A disc is stable if it cannot be flipped by any legal sequence of moves.
    This is a simplified version that checks corner stability and edge stability.
    """
    stable_count = 0
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    
    # Check corner stability
    for r, c in corners:
        if board[r][c] == player:
            stable_count += 1
    
    # Check edge stability (simplified: discs adjacent to stable corners)
    # A more complete implementation would check all possible flip paths
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player:
                # Check if disc is on an edge and adjacent to a stable corner
                if (r == 0 or r == 7 or c == 0 or c == 7):
                    # Check if adjacent to a stable corner
                    is_stable = False
                    for cr, cc in corners:
                        if board[cr][cc] == player:
                            if (r == cr and abs(c - cc) <= 1) or (c == cc and abs(r - cr) <= 1):
                                is_stable = True
                                break
                    if is_stable:
                        stable_count += 1
    
    return stable_count


def evaluate_board_state(board, player):
    """
    Comprehensive board state evaluation heuristic.
    Returns a dictionary with various metrics.
    """
    whites, blacks = count_discs(board)
    white_mobility = get_mobility(board, 1)
    black_mobility = get_mobility(board, -1)
    white_stable = count_stable_discs(board, 1)
    black_stable = count_stable_discs(board, -1)
    
    # Calculate disc difference
    if player == 1:  # White
        disc_diff = whites - blacks
        mobility_diff = white_mobility - black_mobility
        stable_diff = white_stable - black_stable
    else:  # Black
        disc_diff = blacks - whites
        mobility_diff = black_mobility - white_mobility
        stable_diff = black_stable - white_stable
    
    return {
        'disc_count_player': whites if player == 1 else blacks,
        'disc_count_opponent': blacks if player == 1 else whites,
        'disc_difference': disc_diff,
        'mobility_player': white_mobility if player == 1 else black_mobility,
        'mobility_opponent': black_mobility if player == 1 else white_mobility,
        'mobility_difference': mobility_diff,
        'stable_discs_player': white_stable if player == 1 else black_stable,
        'stable_discs_opponent': black_stable if player == 1 else white_stable,
        'stable_discs_difference': stable_diff,
        'total_discs': whites + blacks
    }


def draw_board(board, valid_moves, display=True):
    """Draw the board. Only draws if display is True and pygame is initialized."""
    if not display or not pygame_initialized:
        return
    screen.fill(GREEN)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

            if (row, col) in valid_moves:
                pygame.draw.circle(screen, HIGHLIGHT,
                                   (col * CELL_SIZE + CELL_SIZE // 2,
                                    row * CELL_SIZE + CELL_SIZE // 2), 8)

            if board[row][col] == 1:
                pygame.draw.circle(screen, WHITE,
                                   (col * CELL_SIZE + CELL_SIZE // 2,
                                    row * CELL_SIZE + CELL_SIZE // 2), 30)
            elif board[row][col] == -1:
                pygame.draw.circle(screen, BLACK,
                                   (col * CELL_SIZE + CELL_SIZE // 2,
                                    row * CELL_SIZE + CELL_SIZE // 2), 30)

def create_agent(agent_details, first):
    agent_type = agent_details[0]
    match agent_type:
        case "human":
            return HumanAgent(first)
        case "minimax":
            use_alpha_beta = agent_details[1] if len(agent_details) > 1 else True
            depth = agent_details[2] if len(agent_details) > 2 else 3
            return MinimaxAgent(first, use_alpha_beta, depth)
        case "mcts":
            # Parse MCTS parameters: [iterations, rollout_type, rollout_simulations]
            iterations = agent_details[1] if len(agent_details) > 1 else 500
            rollout_type = agent_details[2] if len(agent_details) > 2 else 'random'
            rollout_simulations = agent_details[3] if len(agent_details) > 3 else 1
            return RandomMCTSAgent(first, iterations=iterations, 
                                  rollout_type=rollout_type, 
                                  rollout_simulations=rollout_simulations)
        case "random":
            return RandomAgent(first)
        case _:
            raise ValueError(f"Unknown agent type: {agent_type}")


# --- Main Game Loop ---
def run_game(environment_data, display=True, output_file="", save_replay=False, replay_file=""):
    """
    Run a single game of Othello.
    
    Args:
        environment_data: Dictionary with agent configurations
        display: Whether to show pygame display
        output_file: Optional file path to write game results
        save_replay: Whether to save replay data
        replay_file: Optional file path to save replay data
    
    Returns:
        Dictionary with game results
    """
    global screen, font
    
    if display:
        init_pygame()
    
    # Load agents from environment data
    black_agent = create_agent(environment_data["agents"][0], -1)
    white_agent = create_agent(environment_data["agents"][1], 1)
    
    board = create_board()
    player = -1  # Black starts
    running = True
    move_count = [0, 0]  # [black_moves, white_moves]
    game_log = []
    move_times = []  # Track timing for each move: [move_number, player, time_seconds, total_discs]
    
    # Record initial board state
    if save_replay:
        whites, blacks = count_discs(board)
        game_log.append({
            'move_number': 0,
            'player': None,
            'move': None,
            'board': board.tolist(),
            'white_score': int(whites),
            'black_score': int(blacks),
            'valid_moves': []
        })
    
    while running:
        valid_moves = get_valid_moves(board, player)
        
        if display:
            draw_board(board, valid_moves, display=True)
            whites, blacks = count_discs(board)
            status_text = f"Turn: {'Black' if player == -1 else 'White'} | W:{whites} B:{blacks}"
            text_surface = font.render(status_text, True, BLACK, GRAY)
            screen.blit(text_surface, (10, WINDOW_SIZE + 10))
            pygame.display.flip()

        # --- Check endgame ---
        if not valid_moves:
            opponent_moves = get_valid_moves(board, -player)
            if not opponent_moves:
                # Game over
                whites, blacks = count_discs(board)
                winner = "White" if whites > blacks else "Black" if blacks > whites else "Draw"
                winner_player = 1 if whites > blacks else -1 if blacks > whites else 0
                
                if display:
                    print(f"Game Over! Winner: {winner}")
                    pygame.time.wait(2000)
                
                # Record final game state
                result = {
                    'winner': winner,
                    'winner_player': winner_player,
                    'white_score': int(whites),
                    'black_score': int(blacks),
                    'total_moves': move_count,
                    'final_board': board.tolist(),
                    'move_times': move_times
                }
                
                # Add final state to replay log
                if save_replay:
                    game_log.append({
                        'move_number': sum(move_count),
                        'player': None,
                        'move': None,
                        'board': board.tolist(),
                        'white_score': int(whites),
                        'black_score': int(blacks),
                        'valid_moves': [],
                        'game_over': True,
                        'winner': winner
                    })
                
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(result, f, indent=2)
                
                # Save replay file
                if save_replay and replay_file:
                    replay_data = {
                        'config': environment_data,
                        'moves': game_log,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    }
                    os.makedirs(os.path.dirname(replay_file) if os.path.dirname(replay_file) else '.', exist_ok=True)
                    with open(replay_file, "w") as f:
                        json.dump(replay_data, f, indent=2)
                    print(f"Replay saved to: {replay_file}")
                
                running = False
                result['replay_data'] = game_log if save_replay else None
                return result
            else:
                # Skip turn
                player *= -1
                continue

        # --- Agent move selection ---
        # Track move computation time
        move_start_time = time.time()
        if player == -1:
            move = black_agent.get_move(board, valid_moves)
        else:
            move = white_agent.get_move(board, valid_moves)
        move_time = time.time() - move_start_time
        
        # Record move timing
        whites, blacks = count_discs(board)
        total_discs = whites + blacks
        move_times.append({
            'move_number': sum(move_count) + 1,
            'player': 'black' if player == -1 else 'white',
            'time_seconds': move_time,
            'total_discs': total_discs,
            'valid_moves_count': len(valid_moves)
        })

        if move:
            # Record move before applying it
            if save_replay:
                whites, blacks = count_discs(board)
                game_log.append({
                    'move_number': sum(move_count) + 1,
                    'player': 'black' if player == -1 else 'white',
                    'move': move,
                    'board_before': board.tolist(),
                    'white_score_before': int(whites),
                    'black_score_before': int(blacks),
                    'valid_moves': valid_moves
                })
            
            board = apply_move(board, move, player)
            move_count[0 if player == -1 else 1] += 1
            
            # Record board state after move
            if save_replay:
                whites, blacks = count_discs(board)
                game_log[-1]['board_after'] = board.tolist()
                game_log[-1]['white_score_after'] = int(whites)
                game_log[-1]['black_score_after'] = int(blacks)
            
            player *= -1

        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
    
    if display:
        pygame.quit()
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Othello Game")
    parser.add_argument(
        "-f", "--filename",
        default="game-environments/example-minimax.json",
        help="Game environment file"
    )
    parser.add_argument(
        "-o", "--outputfile",
        default="",
        help="Output file report"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no display)"
    )
    parser.add_argument(
        "--replay",
        default="",
        help="Save replay file (JSON format)"
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Generate GIF from replay file after game (requires --replay)"
    )
    args = parser.parse_args()
    
    with open(args.filename, "r") as file:
        environment_data = json.load(file)
    
    display = not args.headless
    save_replay = bool(args.replay)
    result = run_game(environment_data, display=display, output_file=args.outputfile, 
                      save_replay=save_replay, replay_file=args.replay)
    
    # Generate GIF if requested
    if args.gif and args.replay and result and result.get('replay_data'):
        try:
            from gif_generator import generate_gif
            gif_file = os.path.splitext(args.replay)[0] + ".gif"
            generate_gif(args.replay, gif_file, duration=500, loop=0)
            print(f"GIF generated: {gif_file}")
        except ImportError:
            print("Warning: Pillow not installed. Cannot generate GIF.")
        except Exception as e:
            print(f"Warning: Could not generate GIF: {e}")


if __name__ == "__main__":
    main()
