# Andy Dao (daoa@msoe.edu)
# Othello Pygame Agents

import random
import pygame
import math
import numpy as np
import sys
import os

CELL_SIZE = 80
BOARD_SIZE = 8  # Default, will be overridden by import if available

# Import game logic from main.py (circular import handled by Python)
try:
    from main import (
        get_valid_moves, apply_move, count_discs, 
        get_mobility, count_stable_discs, BOARD_SIZE
    )
except ImportError:
    # If import fails (e.g., during testing), define stubs
    # These should never be used in normal operation
    def get_valid_moves(board, player):
        raise NotImplementedError("Game logic not imported")
    def apply_move(board, move, player):
        raise NotImplementedError("Game logic not imported")
    def count_discs(board):
        raise NotImplementedError("Game logic not imported")
    def get_mobility(board, player):
        raise NotImplementedError("Game logic not imported")
    def count_stable_discs(board, player):
        raise NotImplementedError("Game logic not imported")


class BaseAgent:
    def __init__(self, player):
        self.player = player

    def get_move(self, board, valid_moves):
        raise NotImplementedError


class HumanAgent(BaseAgent):
    """Handles user mouse input for move selection."""
    def get_move(self, board, valid_moves):
        # In headless mode, human agent cannot work - return None or random move
        try:
            import pygame
            # Check if pygame is initialized and display is available
            if not pygame.get_init():
                # Headless mode - return None (will skip turn)
                return None
        except:
            # Pygame not available - headless mode
            return None
        
        waiting_for_click = True
        while waiting_for_click:
            for event in pygame.event.get():
                # Quit game
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                # Change board
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    row, col = my // CELL_SIZE, mx // CELL_SIZE
                    if (row, col) in valid_moves:
                        waiting_for_click = False
                        return (row, col)
        return None

# MCTS Functions

## MCTS Node Class
class MCTSNode:
    # TODO: add action parameter
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = self.get_actions()

    def get_actions(self):
        # TODO Placeholder for getting valid actions from the state
        return []
    
    def is_terminal(self):
        # TODO Placeholder for terminal state check
        return False
    
    # TODO Check
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    # TODO change to othello
    def check_winner(self):
        """Find winner (1 or 2) or None."""
        for i in range(3):
            if self.state[i][0] == self.state[i][1] == self.state[i][2] != 0:
                return self.state[i][0]
            if self.state[0][i] == self.state[1][i] == self.state[2][i] != 0:
                return self.state[0][i]
        if self.state[0][0] == self.state[1][1] == self.state[2][2] != 0:
            return self.state[0][0]
        if self.state[0][2] == self.state[1][1] == self.state[2][0] != 0:
            return self.state[0][2]
        return None

    ## Selection

    ### TODO Check later
    def expand(self):
            """Add one of the remaining actions as a child."""
            action = self.untried_actions.pop()
            new_state = [row[:] for row in self.state]
            player = self.get_current_player()
            new_state[action[0]][action[1]] = player
            # TODO: action isn't a parameter for a class initialization
            child = MCTSNode(new_state, parent=self, action=action)
            self.children.append(child)
            return child

    def get_current_player(self):
        """Find whose turn it is."""
        x_count = sum(row.count(1) for row in self.state)
        o_count = sum(row.count(2) for row in self.state)
        return 1 if x_count == o_count else 2

    def best_child(self, c=1.4):
        """Select child with best UCB1 score."""
        return max(self.children, key=lambda child:
                    (child.wins / child.visits) +
                    c * math.sqrt(math.log(self.visits) / child.visits))

    def rollout(self):
        """Play random moves until the game ends."""
        state = [row[:] for row in self.state]
        player = self.get_current_player()

        while True:
            winner = self.check_winner_for_state(state)
            if winner: return 1 if winner == 1 else 0

            actions = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0]
            if not actions: return 0.5  # Draw

            move = random.choice(actions)
            state[move[0]][move[1]] = player
            player = 1 if player == 2 else 2

    # TODO: Look into this to see if it's needed
    # def check_winner_for_state(self, state):
    #     """Same winner check for rollout."""
    #     return MCTSNode(state).check_winner()

    def backpropagate(self, result):
        """Update stats up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    ## Search
    ### TODO
def mcts_search(root_state, iterations=500):
    root = MCTSNode(root_state)

    for _ in range(iterations):
        node = root

        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        # Expansion
        if not node.is_terminal():
            node = node.expand()

        # Simulation
        result = node.rollout()

        # Backpropagation
        node.backpropagate(result)


    return root.best_child(c=0).action  # Return best move

## Backpropagation

class RandomMCTSAgent(BaseAgent):
    """Placeholder for MCTS logic — currently plays randomly."""
    # TODO: Implement Monte Carlo Tree Search
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        # TODO: Replace with full Monte Carlo Tree Search
        return random.choice(valid_moves)

class MinimaxAgent(BaseAgent):
    """Minimax agent with alpha-beta pruning for Othello."""
    
    def __init__(self, player, abpruning=True, depth=3):
        super().__init__(player)
        self.abpruning = abpruning
        self.depth = depth
        
        # Heuristic weights (tuned for Othello)
        self.weights = {
            'corner': 25,      # Corner control is very important
            'edge': 5,         # Edge control is valuable
            'mobility': 10,    # Mobility (move options) is important
            'stability': 15,   # Stable discs are valuable
            'disc_count': 1,   # Disc count matters, especially in endgame
            'parity': 0        # Parity can be added later
        }
    
    def evaluate_heuristic(self, board, player):
        """
        Evaluate board state using Othello-specific heuristics.
        Returns a score from the perspective of the given player.
        """
        opponent = -player
        whites, blacks = count_discs(board)
        total_discs = whites + blacks
        
        # Corner control
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        player_corners = sum(1 for r, c in corners if board[r][c] == player)
        opponent_corners = sum(1 for r, c in corners if board[r][c] == opponent)
        corner_score = self.weights['corner'] * (player_corners - opponent_corners)
        
        # Edge control (excluding corners)
        edge_positions = []
        for r in range(BOARD_SIZE):
            if r == 0 or r == 7:
                for c in range(1, 7):  # Exclude corners
                    edge_positions.append((r, c))
            else:
                edge_positions.append((r, 0))
                edge_positions.append((r, 7))
        
        player_edges = sum(1 for r, c in edge_positions if board[r][c] == player)
        opponent_edges = sum(1 for r, c in edge_positions if board[r][c] == opponent)
        edge_score = self.weights['edge'] * (player_edges - opponent_edges)
        
        # Mobility (number of valid moves)
        player_mobility = get_mobility(board, player)
        opponent_mobility = get_mobility(board, opponent)
        mobility_score = self.weights['mobility'] * (player_mobility - opponent_mobility)
        
        # Disc stability
        player_stable = count_stable_discs(board, player)
        opponent_stable = count_stable_discs(board, opponent)
        stability_score = self.weights['stability'] * (player_stable - opponent_stable)
        
        # Disc count (more important in endgame)
        if player == 1:  # White
            disc_diff = whites - blacks
        else:  # Black
            disc_diff = blacks - whites
        
        # Weight disc count more heavily in endgame (when board is mostly filled)
        endgame_weight = self.weights['disc_count'] * (total_discs / 64.0)
        disc_score = endgame_weight * disc_diff
        
        # Combine all heuristics
        total_score = corner_score + edge_score + mobility_score + stability_score + disc_score
        
        return total_score
    
    def is_terminal(self, board):
        """Check if the game is over (no moves for either player)."""
        return (len(get_valid_moves(board, 1)) == 0 and 
                len(get_valid_moves(board, -1)) == 0)
    
    def minimax(self, board, depth, player, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: Current board state
            depth: Remaining search depth
            player: Current player (1 for white, -1 for black)
            alpha: Best value for maximizing player
            beta: Best value for minimizing player
            maximizing_player: True if we're maximizing for self.player
        
        Returns:
            Tuple of (best_score, best_move)
        """
        # Terminal conditions
        if depth == 0 or self.is_terminal(board):
            return self.evaluate_heuristic(board, self.player), None
        
        valid_moves = get_valid_moves(board, player)
        
        # If no valid moves, check opponent
        if not valid_moves:
            opponent_moves = get_valid_moves(board, -player)
            if not opponent_moves:
                # Game over - evaluate final position
                return self.evaluate_heuristic(board, self.player), None
            else:
                # Skip turn - continue with opponent
                return self.minimax(board, depth, -player, alpha, beta, maximizing_player)
        
        best_move = None
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                new_board = apply_move(board, move, player)
                eval_score, _ = self.minimax(
                    new_board, depth - 1, -player, alpha, beta, False
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                if self.abpruning:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
        
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_board = apply_move(board, move, player)
                eval_score, _ = self.minimax(
                    new_board, depth - 1, -player, alpha, beta, True
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                if self.abpruning:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
            
            return min_eval, best_move
    
    def get_move(self, board, valid_moves):
        """Get the best move using minimax search."""
        if not valid_moves:
            return None
        
        # If only one move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Run minimax search
        _, best_move = self.minimax(
            board, 
            self.depth, 
            self.player, 
            float('-inf'), 
            float('inf'), 
            True
        )
        
        # Fallback to first valid move if minimax returns None
        return best_move if best_move is not None else valid_moves[0]

class RandomAgent(BaseAgent):
    """Plays randomly."""
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        return random.choice(valid_moves)