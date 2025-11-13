# Andy Dao (daoa@msoe.edu)
# Othello Pygame Agents

import random
import pygame
import math

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

# MCTS Functions

## MCTS Node Class
class MCTSNode:
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

    # TODO check
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

def check_winner_for_state(self, state):
    """Same winner check for rollout."""
    return MCTSNode(state).check_winner()

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

class RandomMinimaxAgent(BaseAgent):
    """Placeholder for Minimax logic — currently plays randomly."""
    # TODO: Implement Minimax algorithm
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        # TODO: Replace with actual Minimax search
        return random.choice(valid_moves)

