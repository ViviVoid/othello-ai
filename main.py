# Andy Dao (daoa@msoe.edu)
# Othello Pygame

import pygame
import argparse
import json
import sys
import numpy as np
from agent import HumanAgent, RandomMCTSAgent, RandomMinimaxAgent

# --- Game constants ---
BOARD_SIZE = 8
CELL_SIZE = 80
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE
GREEN = (0, 128, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
HIGHLIGHT = (200, 200, 0)

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
pygame.display.set_caption("Othello (Reversi)")
font = pygame.font.SysFont(None, 36)


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
    whites = np.sum(board == 1)
    blacks = np.sum(board == -1)
    return whites, blacks


def draw_board(board, valid_moves):
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


# --- Main Game Loop ---
def main():
    parser = argparse.ArgumentParser(description="Othello Game")
    parser.add_argument(
        "-f", "--filename",
        default="example-minimax.json",
        help="Game environment file"
    )
    parser.add_argument(
        "-o", "--outputfile",
        default="",
        help="Output file report"
    )
    args = parser.parse_args()
    environment_data = []
    with open(args.filename, "r") as file:
        environment_data = json.load(file) # Load environment data
    
    if (args.outputfile != ""):
        with open(args.outputfile, "w") as file:
            # file.write("Hello World")
            print("Writing to output file...")
    board = create_board()
    player = -1  # Black starts
    running = True

    # --- Choose agents here ---
    # Examples:
    black_agent = HumanAgent(-1)
    white_agent = RandomMCTSAgent(1)
    # white_agent = RandomMinimaxAgent(1)

    while running:
        valid_moves = get_valid_moves(board, player)
        draw_board(board, valid_moves)
        whites, blacks = count_discs(board)
        status_text = f"Turn: {'Black' if player == -1 else 'White'} | W:{whites} B:{blacks}"
        text_surface = font.render(status_text, True, BLACK, GRAY)
        screen.blit(text_surface, (10, WINDOW_SIZE + 10))
        pygame.display.flip()

        # --- Check endgame ---
        if not valid_moves:
            if not get_valid_moves(board, -player):
                winner = "White" if whites > blacks else "Black" if blacks > whites else "Draw"
                print(f"Game Over! Winner: {winner}")
                pygame.time.wait(2000)
                running = False
                continue
            else:
                player *= -1
                continue

        # --- Agent move selection ---
        if player == -1:
            move = black_agent.get_move(board, valid_moves)
        else:
            move = white_agent.get_move(board, valid_moves)

        if move:
            board = apply_move(board, move, player)
            player *= -1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    pygame.quit()


if __name__ == "__main__":
    main()
