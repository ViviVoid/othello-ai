# Andy Dao (daoa@msoe.edu)
# Othello Pygame Agents

import random
import pygame

CELL_SIZE = 80


class BaseAgent:
    def __init__(self, player):
        self.player = player

    def get_move(self, board, valid_moves):
        raise NotImplementedError


class HumanAgent(BaseAgent):
    """Handles user mouse input for move selection."""
    def get_move(self, board, valid_moves):
        waiting_for_click = True
        while waiting_for_click:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    row, col = my // CELL_SIZE, mx // CELL_SIZE
                    if (row, col) in valid_moves:
                        waiting_for_click = False
                        return (row, col)
        return None


class RandomMCTSAgent(BaseAgent):
    """Placeholder for MCTS logic — currently plays randomly."""
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        # TODO: Replace with full Monte Carlo Tree Search
        return random.choice(valid_moves)


class RandomMinimaxAgent(BaseAgent):
    """Placeholder for Minimax logic — currently plays randomly."""
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        # TODO: Replace with actual Minimax search
        return random.choice(valid_moves)
