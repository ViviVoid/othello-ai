# Andy Dao (daoa@msoe.edu)
# Sierra Andrews (andrewss@msoe.edu)
# Othello Pygame Agents

import random
import pygame
import math
import numpy as np
import sys
import os

#from main import get_valid_moves, count_discs, apply_move

CELL_SIZE = 80
BOARD_SIZE = 8  # Default, will be overridden by import if available

# Lazy import helper to avoid circular imports
_game_functions = None

def _get_game_functions():
    """Lazy import of game logic functions from main.py to avoid circular imports."""
    global _game_functions
    if _game_functions is None:
        try:
            import main
            _game_functions = {
                'get_valid_moves': main.get_valid_moves,
                'apply_move': main.apply_move,
                'count_discs': main.count_discs,
                'get_mobility': main.get_mobility,
                'count_stable_discs': main.count_stable_discs,
                'BOARD_SIZE': main.BOARD_SIZE
            }
        except (ImportError, AttributeError):
            raise ImportError(
                "Could not import game logic from main.py. "
                "Make sure main.py is available and contains the required functions."
            )
    return _game_functions


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

# TODO: keep player variable?
# TODO: Fix indentation errors
## MCTS Node Class
class MCTSNode:
    def __init__(self, state, player, parent=None, action=None):
        # All possible directions
        self.DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        # Ensure state is a list of lists (not numpy array)
        if isinstance(state, np.ndarray):
            self.state = state.tolist()
        elif isinstance(state, list):
            self.state = [list(row) for row in state]
        else:
            self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.action = action
        self.player = player
        self.untried_actions = self.get_actions()



    def on_board(self, r, c):
        """Ensures players stay on board"""
        return 0 <= r < 8 and 0 <= c < 8

    def game_over(self):
        """True if neither player has a valid move."""
        player_moves = self.get_actions()
        opponent_moves = MCTSNode(self.state, -self.player).get_actions()
        return not player_moves and not opponent_moves

    def get_actions(self):
        """Get valid actions from the state"""
        # opponent = 2 if self.player == 1 else 1
        opponent = -self.player
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
            if isinstance(self.state, np.ndarray):
                return self.state.copy().tolist()
            return [list(row) for row in self.state]

        r, c = action
        opponent = -self.player

        # Make copy so original board is unchanged
        if isinstance(self.state, np.ndarray):
            new_board = self.state.copy().tolist()
        else:
            new_board = [list(row) for row in self.state]

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
        # original line: new_state = apply_action(self.state, self.player, move)
        # new_state = self.state.apply_action(self.player, move)
        # Complete said move
        new_state = self.apply_action(move)
        # Change players
        # next_player = 2 if self.player == 1 else 1
        next_player = -self.player
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
        if not self.children:
            return None
        # Safety check: if visits is 0, use a default value
        return max(self.children, key=lambda child:
                    (child.wins / child.visits if child.visits > 0 else 0) +
                    c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1)))


    def rollout(self, root_player=None):
        """Play random moves until the game ends.
        
        Args:
            root_player: The player at the root of the MCTS tree (for evaluation).
                        If None, uses self.player (for backward compatibility).
        """
        # Stops circular crashing
        from main import get_valid_moves, apply_move, count_discs
        
        # Convert state to numpy array if it's a list
        if isinstance(self.state, list):
            state = np.array(self.state)
        else:
            state = np.copy(self.state)
        
        # Get current player (the player whose turn it is at this node)
        player = self.player
        
        # Determine root player for evaluation (the player whose perspective we evaluate from)
        # If not provided, trace back to root
        if root_player is None:
            node = self
            while node.parent is not None:
                node = node.parent
            # Root player is the parent's opponent (since root's children are after root's move)
            # Actually, root's player IS the root player, so we need to go up one more level
            # Wait, let's think: root node has player = root_player
            # After root makes a move, child has player = -root_player
            # So if we're at a child, parent.player is root_player
            # But we want the original root player...
            # Actually, simpler: store root player in node, or pass it down
            # For now, use a simpler approach: evaluate from the perspective of the node that started the rollout
            root_player = self.player if self.parent is None else self._get_root_player()
        
        while True:
            # Get possible moves
            moves = get_valid_moves(state, player)

            # if no viable moves
            if not moves:
                op_moves = get_valid_moves(state, -player)
                # if opponent has no viable moves, game ends
                if not op_moves:
                    # Count discs
                    w, b = count_discs(state)
                    # Find winner and return numerical representation from root player's perspective
                    # Return 1 if root player wins, 0 if root player loses, 0.5 for draw
                    if root_player == 1:  # Root player is white
                        return 1.0 if w > b else 0.0 if b > w else 0.5
                    else:  # Root player is black
                        return 1.0 if b > w else 0.0 if w > b else 0.5

                # Skip turn
                player = -player
                continue

            # Pick a random move for the simulation portion
            move = random.choice(moves)
            # apply move to board
            state = apply_move(state, move, player)
            # Change player
            player = -player
    
    def _get_root_player(self):
        """Trace back to root to find the root player."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node.player




    def backpropagate(self, result):
        """Update stats up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

## Search
### DONE (theoretically)
def mcts_search(root_state, root_player, iterations=500):
    """
    MCTS search to find best move.
    
    Args:
        root_state: Current board state
        root_player: Player whose turn it is (1 for white, -1 for black)
        iterations: Number of MCTS iterations
    """
    root = MCTSNode(root_state, root_player)

    for _ in range(iterations):
        node = root

        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            next_node = node.best_child()
            if next_node is None:
                break  # Can't select further, break out of selection
            node = next_node

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            expanded_node = node.expand()
            if expanded_node is not None:
                node = expanded_node

        # Simulation - pass root player so evaluation is from correct perspective
        result = node.rollout(root_player=root_player)

        # Backpropagation
        node.backpropagate(result)

    # Return best move
    if not root.children:
        # Edge case: no children (shouldn't happen if there are valid moves)
        return None
    best_child_node = root.best_child(c=0)
    if best_child_node is None:
        return None
    return best_child_node.action


class RandomMCTSAgent(BaseAgent):
    """MCTS agent using Monte Carlo Tree Search."""
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        
        # Convert board to list of lists (handle numpy arrays)
        if isinstance(board, np.ndarray):
            state_copy = board.tolist()
        else:
            state_copy = [list(row) for row in board]

        # Find best action - pass self.player so MCTS knows whose turn it is
        best_move = mcts_search(state_copy, self.player, iterations=500)

        # If MCTS returns None or invalid move, fall back to random
        if best_move is None or best_move not in valid_moves:
            return random.choice(valid_moves)

        return best_move

class MinimaxAgent(BaseAgent):
    """Minimax agent with alpha-beta pruning for Othello."""
    
    def __init__(self, player, abpruning=True, depth=3):
        super().__init__(player)
        self.abpruning = abpruning
        self.depth = depth
        
        # Lazy load game functions on first use
        self._game_funcs = None
        
        # Pre-compute edge positions (static, so compute once)
        self.edge_positions = []
        BOARD_SIZE = 8
        for r in range(BOARD_SIZE):
            if r == 0 or r == 7:
                for c in range(1, 7):  # Exclude corners
                    self.edge_positions.append((r, c))
            else:
                self.edge_positions.append((r, 0))
                self.edge_positions.append((r, 7))
        
        # Heuristic weights (tuned for Othello)
        self.weights = {
            'corner': 25,      # Corner control is very important
            'edge': 5,         # Edge control is valuable
            'mobility': 10,    # Mobility (move options) is important
            'stability': 15,   # Stable discs are valuable
            'disc_count': 1,   # Disc count matters, especially in endgame
            'parity': 0        # Parity can be added later
        }
    
    def _load_game_functions(self):
        """Lazy load game functions when needed."""
        if self._game_funcs is None:
            self._game_funcs = _get_game_functions()
        return self._game_funcs
    
    def _quick_evaluate_move(self, board, move, player):
        """Quick evaluation of a single move for move ordering."""
        # Simple heuristic: prefer corners, then edges
        r, c = move
        if (r, c) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
            return 100  # Corners are best
        elif (r, c) in self.edge_positions:
            return 10   # Edges are good
        else:
            return 1    # Interior moves
    
    def _order_moves(self, board, moves, player):
        """Order moves by heuristic value to improve alpha-beta pruning."""
        # Sort moves by quick evaluation (best first)
        return sorted(moves, key=lambda m: self._quick_evaluate_move(board, m, player), reverse=True)
    
    def evaluate_heuristic(self, board, player):
        """
        Evaluate board state using Othello-specific heuristics.
        Returns a score from the perspective of the given player.
        Optimized version that avoids redundant calculations.
        """
        game_funcs = self._load_game_functions()
        count_discs = game_funcs['count_discs']
        get_mobility = game_funcs['get_mobility']
        count_stable_discs = game_funcs['count_stable_discs']
        
        opponent = -player
        whites, blacks = count_discs(board)
        total_discs = whites + blacks
        
        # Corner control (pre-computed corners list)
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        player_corners = sum(1 for r, c in corners if board[r][c] == player)
        opponent_corners = sum(1 for r, c in corners if board[r][c] == opponent)
        corner_score = self.weights['corner'] * (player_corners - opponent_corners)
        
        # Edge control (using pre-computed edge positions)
        player_edges = sum(1 for r, c in self.edge_positions if board[r][c] == player)
        opponent_edges = sum(1 for r, c in self.edge_positions if board[r][c] == opponent)
        edge_score = self.weights['edge'] * (player_edges - opponent_edges)
        
        # Mobility (number of valid moves) - only compute if needed
        player_mobility = get_mobility(board, player)
        opponent_mobility = get_mobility(board, opponent)
        mobility_score = self.weights['mobility'] * (player_mobility - opponent_mobility)
        
        # Disc stability - can be expensive, skip in early game
        if total_discs > 20:  # Only compute stability after some discs are placed
            player_stable = count_stable_discs(board, player)
            opponent_stable = count_stable_discs(board, opponent)
            stability_score = self.weights['stability'] * (player_stable - opponent_stable)
        else:
            stability_score = 0
        
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
        game_funcs = self._load_game_functions()
        get_valid_moves = game_funcs['get_valid_moves']
        return (len(get_valid_moves(board, 1)) == 0 and 
                len(get_valid_moves(board, -1)) == 0)
    
    def minimax(self, board, depth, player, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning and move ordering.
        
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
        game_funcs = self._load_game_functions()
        get_valid_moves = game_funcs['get_valid_moves']
        apply_move = game_funcs['apply_move']
        
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
        
        # Order moves for better alpha-beta pruning (best moves first)
        ordered_moves = self._order_moves(board, valid_moves, player)
        
        best_move = None
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in ordered_moves:
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
            for move in ordered_moves:
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