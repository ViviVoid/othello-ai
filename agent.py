# Andy Dao (daoa@msoe.edu)
# Sierra Andrews (andrewss@msoe.edu)
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

# TODO: keep player variable?
# TODO: Fix indentation errors
## MCTS Node Class
class MCTSNode:
    def __init__(self, state, player, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.action = action
        self.player = player
        self.untried_actions = self.get_actions()
        # All possible directions
        self.DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


    def on_board(self, r, c):
        """Ensures players stay on board"""
        return 0 <= r < 8 and 0 <= c < 8

    def game_over(self):
        """True if neither player has a valid move."""
        return not self.get_actions() and not self.get_actions()

    def get_actions(self):
        """Get valid actions from the state"""
        opponent = 2 if self.player == 1 else 1
        valid_moves = []

        # loop through every square
        for r in range(8):
            for c in range(8):
                # Skip squares that aren't empty
                if self.state[r][c] != 0:
                    continue

                move_is_valid = False

                # for each row and col in a direction (r, c), check if placing a piece would flip opponent
                for dr, dc in self.DIRECTIONS:
                    # Step in 1 direction (immediate neighbor)
                    rr, cc = r + dr, c + dc

                    # If the immediate neighbor is not an opponent, keep checking
                    if not self.on_board(rr, cc) or self.state[rr][cc] != opponent:
                        continue
                    # Keep traveling until non-opponent square is found
                    while self.on_board(rr, cc) and self.state[rr][cc] == opponent:
                        rr += dr
                        cc += dc
                    # If we end line on current player piece, we can legally flip
                    if self.on_board(rr, cc) and self.state[rr][cc] == self.player:
                        move_is_valid = True
                        break
                # After all directions are checked, add move if valid
                if move_is_valid:
                    valid_moves.append((r, c))

        return valid_moves


    def is_terminal(self):
        # TODO Placeholder for terminal state check
        return self.game_over()

    # TODO Check (DONE)
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    # TODO: find way to check board
    def check_winner(self):
        """Return 1 or 2 or None for tie."""
        # would self.state work?
        p1 = sum(row.count(1) for row in self.state)
        p2 = sum(row.count(2) for row in self.state)
        if p1 > p2:
            return 1
        elif p2 > p1:
            return 2
        return None

    def apply_action(self, action):
        """
            Returns a NEW board after placing a piece for `player` at `move`
            and flipping all appropriate opponent discs.
            """
        if action is None:
            # Don't do anything and return copy of the board
            return [row[:] for row in self.state]

        r, c = action
        opponent = 2 if self.player == 1 else 1

        # Make copy so original board is unchanged
        new_board = [row[:] for row in self.state]

        # Place player's piece
        new_board[r][c] = self.player

        # For each direction, check for flippable discs
        for dr, dc in self.DIRECTIONS:
            rr, cc = r + dr, c + dc
            path = []

            # First step must be opponent discs (or nothing flips)
            while self.on_board(rr, cc) and new_board[rr][cc] == opponent:
                path.append((rr, cc))
                rr += dr
                cc += dc

            # After traveling opponent pieces, next must be player's piece
            if self.on_board(rr, cc) and new_board[rr][cc] == self.player:
                # Valid flipping line => flip everything in path
                for fr, fc in path:
                    new_board[fr][fc] = self.player

        # Return new board state
        return new_board


    ## Selection

    ### TODO Check later
    def expand(self):
        """Add one of the remaining actions as a child."""
        # # Expand one move
        # action = self.untried_actions.pop()
        # # Copy state to new board
        # new_state = [row[:] for row in self.state]
        # # grab current player
        # player = self.get_current_player()
        # #
        # new_state[action[0]][action[1]] = player
        # child = MCTSNode(new_state, parent=self, action=action)
        # self.children.append(child)
        # return child

        # Selects one move to make
        move = self.untried_actions.pop()
        # Complete said move
        # TODO: Fix error
        new_state = apply_action(self.state, self.player, move)
        # Change players
        next_player = 2 if self.player == 1 else 1
        # Create new MCTS for new state
        child = MCTSNode(new_state, next_player, parent=self, action=move)
        # Add new node to child list
        self.children.append(child)
        # return new node to continue selection + simulation
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
        # Copy board so we don't overwrite the original
        state = [row[:] for row in self.state]
        # Get current player
        player = self.get_current_player()

        while True:
            # TODO: Check replacement code is equivalent?
            # winner = self.check_winner_for_state(state)
            # Determine if there is a winner
            winner = MCTSNode(state).check_winner()
            # Determine who winner is if there is one
            if winner: return 1 if winner == 1 else 0

            # Find available moves
            actions = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0]
            # If there are no available moves return a draw
            if not actions: return 0.5

            # Determine random move
            move = random.choice(actions)
            # Make determined move
            state[move[0]][move[1]] = player
            # Switch players
            player = 1 if player == 2 else 2


    def backpropagate(self, result):
        """Update stats up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

## Search
### DONE
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

    # Return best move
    # TODO: Fix
    return best_child(root, c=0).action


class RandomMCTSAgent(BaseAgent):
    """Placeholder for MCTS logic — currently plays randomly."""
    # TODO: Implement Monte Carlo Tree Search
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        # TODO: Replace with full Monte Carlo Tree Search
        return random.choice(valid_moves)

class MinimaxAgent(BaseAgent):
    """Placeholder for Minimax logic — currently plays randomly."""
    # TODO: Implement Minimax algorithm
    def __init__(self, player, abpruning=False, depth=1):
        super().__init__(player)
        self.abpruning = abpruning
        self.depth = depth
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        # TODO: Replace with actual Minimax search
        return random.choice(valid_moves)

class RandomAgent(BaseAgent):
    """Plays randomly."""
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        return random.choice(valid_moves)