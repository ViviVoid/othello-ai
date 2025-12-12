"""
Othello AI Agent Implementations

This module provides implementations of various AI agents for playing Othello,
including HumanAgent (for interactive play), RandomAgent (random move selection),
MinimaxAgent (minimax algorithm with alpha-beta pruning and heuristic evaluation),
and RandomMCTSAgent (Monte Carlo Tree Search with tree reuse).

The module also contains the MCTSNode class and MCTS search functions for
implementing Monte Carlo Tree Search with various rollout strategies.

Key Features:
- Multiple agent types with different strategies and complexity levels
- Minimax with alpha-beta pruning and Othello-specific heuristics
- MCTS with configurable iterations, rollout strategies, and tree reuse
- Lazy import of game functions to avoid circular dependencies

Authors: 
    Andy Dao (daoa@msoe.edu)
    Sierra Andrews (andrewss@msoe.edu)
"""

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
    """
    Lazy import of game logic functions from main.py to avoid circular imports.
    
    This function provides access to game logic functions (get_valid_moves, apply_move,
    count_discs, etc.) without creating circular import dependencies. Functions are
    loaded on first access and cached for subsequent use.
    
    Returns:
        dict: Dictionary mapping function names to function objects:
            - 'get_valid_moves': Function to get valid moves
            - 'apply_move': Function to apply a move to board
            - 'count_discs': Function to count discs
            - 'get_mobility': Function to calculate mobility
            - 'count_stable_discs': Function to count stable discs
            - 'BOARD_SIZE': Board size constant
    
    Raises:
        ImportError: If main.py cannot be imported or required functions are missing
    """
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
    """
    Base class for all Othello agents.
    
    This abstract base class defines the interface that all agent implementations
    must follow. Subclasses must implement the get_move method to provide move
    selection logic.
    
    Attributes:
        player (int): Player value (1 for white, -1 for black)
    """
    
    def __init__(self, player):
        """
        Initialize the agent with a player value.
        
        Args:
            player (int): Player value (1 for white, -1 for black)
        """
        self.player = player

    def get_move(self, board, valid_moves):
        """
        Select a move from the list of valid moves.
        
        This method must be implemented by subclasses to provide agent-specific
        move selection logic.
        
        Args:
            board (numpy.ndarray): Current board state
            valid_moves (list): List of (row, col) tuples representing valid moves
        
        Returns:
            tuple: (row, col) tuple representing the selected move, or None if no move
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


class HumanAgent(BaseAgent):
    """
    Agent that handles user mouse input for move selection.
    
    This agent waits for the user to click on a valid move position using the
    mouse. It requires pygame to be initialized and display mode to be enabled.
    In headless mode, it returns None (will skip turn).
    """
    
    def get_move(self, board, valid_moves):
        """
        Wait for user mouse click to select a move.
        
        This method blocks until the user clicks on a valid move position.
        It handles pygame events and returns the clicked position if it's a
        valid move.
        
        Args:
            board (numpy.ndarray): Current board state (not used for human input)
            valid_moves (list): List of (row, col) tuples representing valid moves
        
        Returns:
            tuple: (row, col) tuple of the clicked position if valid, None otherwise
                Returns None if pygame is not initialized (headless mode)
        """
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

class MCTSNode:
    """
    Node class for Monte Carlo Tree Search.
    
    Each node represents a game state in the MCTS tree. Nodes track statistics
    (visits, wins) and maintain connections to parent and child nodes. The class
    implements the core MCTS operations: selection, expansion, simulation, and
    backpropagation.
    
    Attributes:
        state (list): Board state as list of lists (converted from numpy array)
        parent (MCTSNode): Parent node in the tree, None for root
        children (list): List of child MCTSNode objects
        visits (int): Number of times this node has been visited
        wins (float): Accumulated win scores from simulations
        action (tuple): The action (move) that led to this state from parent
        player (int): Player whose turn it is at this state (1 for white, -1 for black)
        rollout_type (str): Type of rollout strategy ('random' or 'minimax')
        rollout_simulations (int): Number of rollout simulations to average
        untried_actions (list): List of actions not yet expanded as children
    """
    
    def __init__(self, state, player, parent=None, action=None, rollout_type='random', rollout_simulations=1):
        """
        Initialize an MCTS node.
        
        Args:
            state: Board state (numpy array or list of lists)
            player (int): Player whose turn it is (1 for white, -1 for black)
            parent (MCTSNode, optional): Parent node. Defaults to None.
            action (tuple, optional): Action that led to this state. Defaults to None.
            rollout_type (str, optional): Rollout strategy ('random' or 'minimax'). Defaults to 'random'.
            rollout_simulations (int, optional): Number of simulations per rollout. Defaults to 1.
        """
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
        self.rollout_type = rollout_type  # 'random' or 'minimax'
        self.rollout_simulations = rollout_simulations  # Number of simulations for averaging
        self.untried_actions = self.get_actions()



    def on_board(self, r, c):
        """
        Check if coordinates are within board boundaries.
        
        Args:
            r (int): Row index
            c (int): Column index
        
        Returns:
            bool: True if coordinates are valid, False otherwise
        """
        return 0 <= r < 8 and 0 <= c < 8

    def game_over(self):
        """
        Check if the game has ended (neither player has valid moves).
        
        Returns:
            bool: True if game is over, False otherwise
        """
        player_moves = self.get_actions()
        opponent_moves = MCTSNode(self.state, -self.player).get_actions()
        return not player_moves and not opponent_moves

    def get_actions(self):
        """
        Get valid moves (actions) from the current board state.
        
        Checks all empty positions on the board and determines if placing a disc
        at that position would flip at least one opponent disc in any direction.
        
        Returns:
            list: List of (row, col) tuples representing valid moves
        """
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
        """
        Check if this node represents a terminal (game over) state.
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.game_over()

    def is_fully_expanded(self):
        """
        Check if all possible actions from this state have been expanded.
        
        Returns:
            bool: True if all actions have been tried, False otherwise
        """
        return len(self.untried_actions) == 0

    def check_winner(self):
        """
        Determine the winner based on disc count.
        
        Note: This implementation uses player values 1 and 2, but the game
        uses 1 (white) and -1 (black). This method may need adjustment.
        
        Returns:
            int: 1 if player 1 wins, 2 if player 2 wins, None for tie
        """
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
        Apply an action (move) to the current state and return a new board.
        
        Places the player's disc at the action position and flips all opponent
        discs that are sandwiched between the new disc and an existing player
        disc in any direction. Returns a new board state without modifying
        the current state.
        
        Args:
            action (tuple): (row, col) position to place the disc, or None for pass
        
        Returns:
            list: New board state as list of lists with the move applied
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


    def expand(self):
        """
        Expand the tree by adding a new child node for an untried action.
        
        Selects one action from untried_actions, applies it to create a new
        game state, and creates a child MCTSNode with the resulting state.
        The child inherits rollout settings from the parent.
        
        Returns:
            MCTSNode: The newly created child node
        """
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
        new_state = self.apply_action(move)
        # Change players
        next_player = -self.player
        # Create new MCTS node for new state, inheriting rollout settings from parent
        child = MCTSNode(new_state, next_player, parent=self, action=move,
                        rollout_type=self.rollout_type, rollout_simulations=self.rollout_simulations)
        # Add new node to child list
        self.children.append(child)
        # return new node to continue selection + simulation
        return child

    def get_current_player(self):
        """
        Determine whose turn it is based on disc count.
        
        Note: This method uses player values 1 and 2, which may not match
        the game's use of 1 and -1. Consider using self.player instead.
        
        Returns:
            int: 1 if player 1's turn, 2 if player 2's turn
        """
        x_count = sum(row.count(1) for row in self.state)
        o_count = sum(row.count(2) for row in self.state)
        return 1 if x_count == o_count else 2

    def best_child(self, c=1.4):
        """
        Select the child node with the best UCB1 (Upper Confidence Bound) score.
        
        UCB1 balances exploitation (choosing moves with high win rate) and
        exploration (trying moves that haven't been explored much). The formula
        is: win_rate + c * sqrt(ln(parent_visits) / child_visits)
        
        Args:
            c (float): Exploration constant (default: 1.4). Higher values favor exploration.
        
        Returns:
            MCTSNode: Child node with highest UCB1 score, or None if no children
        """
        if not self.children:
            return None
        # Safety check: if visits is 0, use a default value
        return max(self.children, key=lambda child:
                    (child.wins / child.visits if child.visits > 0 else 0) +
                    c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1)))


    def _minimax_rollout_move(self, state, player, root_player):
        """
        Use depth-1 minimax (greedy evaluation) to select a move during rollout.
        
        This method evaluates all valid moves for the current player using a
        simple heuristic (disc difference) and selects the best move from the
        root player's perspective. This provides smarter rollout behavior than
        random selection.
        
        Args:
            state (numpy.ndarray): Current board state
            player (int): Player whose turn it is (1 for white, -1 for black)
            root_player (int): Player at the root of the MCTS tree (for evaluation)
        
        Returns:
            tuple: (row, col) tuple representing the best move, or random move if evaluation fails
        """
        from main import get_valid_moves, apply_move, count_discs, evaluate_board_state
        
        # Ensure state is numpy array
        if isinstance(state, list):
            state = np.array(state)
        
        valid_moves = get_valid_moves(state, player)
        if not valid_moves:
            return None
        
        # Use depth-1 minimax (greedy evaluation)
        best_move = None
        best_score = float('-inf') if player == root_player else float('inf')
        
        for move in valid_moves:
            new_state = apply_move(state, move, player)
            # Evaluate from root player's perspective
            score = evaluate_board_state(new_state, root_player)['disc_difference']
            
            if player == root_player:
                # Maximizing for root player
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                # Minimizing for opponent
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move if best_move is not None else random.choice(valid_moves)
    
    def rollout(self, root_player=None):
        """
        Perform rollout simulation(s) from this node to game end.
        
        Runs one or more simulations (based on rollout_simulations) from the
        current state to the end of the game using the specified rollout strategy
        (random or minimax). Returns the average result across all simulations.
        
        The result is a value between 0 and 1: 1.0 if root_player wins,
        0.0 if root_player loses, 0.5 for a draw.
        
        Args:
            root_player (int, optional): Player at the root of MCTS tree (for evaluation).
                                        If None, traces back to root to find it.
        
        Returns:
            float: Average win rate (0.0 to 1.0) from root_player's perspective
        """
        # Stops circular crashing
        from main import get_valid_moves, apply_move, count_discs
        
        # Run multiple simulations and average results if rollout_simulations > 1
        results = []
        for _ in range(self.rollout_simulations):
            results.append(self._single_rollout(root_player))
        
        # Return average of all simulations
        return sum(results) / len(results) if results else 0.5
    
    def _single_rollout(self, root_player=None):
        """
        Perform a single rollout simulation from this node to game end.
        
        Simulates play from the current state until the game ends, using the
        rollout strategy (random or minimax) to select moves. Returns 1.0 if
        root_player wins, 0.0 if root_player loses, 0.5 for a draw.
        
        Args:
            root_player (int, optional): Player at root of MCTS tree. If None, finds it.
        
        Returns:
            float: Win result (1.0 = win, 0.0 = loss, 0.5 = draw) for root_player
        """
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

            # Select move based on rollout type
            if self.rollout_type == 'minimax':
                move = self._minimax_rollout_move(state, player, root_player)
            else:  # 'random'
                move = random.choice(moves)
            
            # apply move to board
            state = apply_move(state, move, player)
            # Change player
            player = -player
    
    def _get_root_player(self):
        """
        Trace back through parent nodes to find the root player.
        
        Returns:
            int: Player value at the root of the MCTS tree
        """
        node = self
        while node.parent is not None:
            node = node.parent
        return node.player




    def backpropagate(self, result):
        """
        Backpropagate simulation results up the tree.
        
        Updates visit count and win count for this node and recursively
        backpropagates to all ancestor nodes. This maintains statistics
        for the UCB1 selection algorithm.
        
        Args:
            result (float): Win result from rollout (0.0 to 1.0)
        """
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

## Search

def mcts_search(root_state, root_player, iterations=500, rollout_type='random', rollout_simulations=1):
    """
    Perform MCTS search to find the best move (creates new tree).
    
    This function creates a new MCTS tree from the root state, performs the
    specified number of iterations (selection, expansion, simulation, backpropagation),
    and returns the best move based on visit counts.
    
    Args:
        root_state: Current board state (numpy array or list of lists)
        root_player (int): Player whose turn it is (1 for white, -1 for black)
        iterations (int, optional): Number of MCTS iterations. Defaults to 500.
        rollout_type (str, optional): Rollout strategy ('random' or 'minimax'). Defaults to 'random'.
        rollout_simulations (int, optional): Number of simulations per rollout. Defaults to 1.
    
    Returns:
        tuple: (row, col) tuple representing the best move, or None if no valid moves
    """
    root = MCTSNode(root_state, root_player, rollout_type=rollout_type, rollout_simulations=rollout_simulations)
    return mcts_search_with_tree(root, iterations)

def mcts_search_with_tree(root, iterations=500):
    """
    Perform MCTS search using an existing tree root (for tree reuse).
    
    This function performs MCTS iterations starting from an existing root node,
    allowing for tree reuse between moves. This is more efficient than creating
    a new tree each time, as the tree from the previous move can be reused for
    the opponent's move.
    
    Args:
        root (MCTSNode): Root node of the MCTS tree (may be reused from previous search)
        iterations (int, optional): Number of MCTS iterations. Defaults to 500.
    
    Returns:
        tuple: (row, col) tuple representing the best move, or None if no valid moves
    """
    root_player = root.player

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
    """
    MCTS agent using Monte Carlo Tree Search with tree reuse.
    
    This agent uses Monte Carlo Tree Search to select moves. It maintains the
    MCTS tree between moves to improve efficiency by reusing previously computed
    search results. Supports configurable iterations, rollout strategies, and
    multiple rollout simulations.
    
    Attributes:
        player (int): Player value (1 for white, -1 for black)
        tree_root (MCTSNode): Root of the MCTS tree (reused between moves)
        last_board_state: Previous board state for tree reuse optimization
        iterations (int): Number of MCTS iterations per move
        rollout_type (str): Rollout strategy ('random' or 'minimax')
        rollout_simulations (int): Number of rollout simulations to average
    """
    
    def __init__(self, player, iterations=500, rollout_type='random', rollout_simulations=1):
        """
        Initialize the MCTS agent.
        
        Args:
            player (int): Player value (1 for white, -1 for black)
            iterations (int, optional): Number of MCTS iterations per move. Defaults to 500.
            rollout_type (str, optional): Rollout strategy ('random' or 'minimax'). Defaults to 'random'.
            rollout_simulations (int, optional): Number of simulations per rollout. Defaults to 1.
        """
        super().__init__(player)
        self.tree_root = None  # Store the MCTS tree between moves
        self.last_board_state = None  # Track the last board state we searched
        self.iterations = iterations
        self.rollout_type = rollout_type  # 'random' or 'minimax'
        self.rollout_simulations = rollout_simulations  # Number of simulations per rollout
    
    def _states_equal(self, state1, state2):
        """
        Check if two board states are equal.
        
        Compares board states, handling both numpy arrays and lists.
        
        Args:
            state1: First board state (numpy array or list)
            state2: Second board state (numpy array or list)
        
        Returns:
            bool: True if states are equal, False otherwise
        """
        if isinstance(state1, np.ndarray):
            state1 = state1.tolist()
        if isinstance(state2, np.ndarray):
            state2 = state2.tolist()
        return state1 == state2
    
    def _find_child_by_state(self, node, target_state):
        """
        Find a child node that matches the target state.
        
        Args:
            node (MCTSNode): Node to search children of
            target_state: Board state to match
        
        Returns:
            MCTSNode: Matching child node, or None if not found
        """
        for child in node.children:
            if self._states_equal(child.state, target_state):
                return child
        return None
    
    def _find_node_by_state(self, node, target_state, max_depth=3):
        """
        Recursively find a node matching the target state (limited depth for efficiency).
        
        Searches the tree starting from the given node to find a node with a matching
        state. Limits search depth to avoid performance issues.
        
        Args:
            node (MCTSNode): Starting node for search
            target_state: Board state to match
            max_depth (int, optional): Maximum depth to search. Defaults to 3.
        
        Returns:
            MCTSNode: Matching node, or None if not found
        """
        if self._states_equal(node.state, target_state):
            return node
        
        if max_depth <= 0:
            return None
        
        # Search in children
        for child in node.children:
            found = self._find_node_by_state(child, target_state, max_depth - 1)
            if found is not None:
                return found
        
        return None
    
    def _update_tree_root(self, new_state, new_player):
        """
        Update the tree root to the new game state, reusing subtree if possible.
        
        Attempts to find the new game state in the existing tree (within limited depth)
        to reuse previously computed search results. If found, the matching node becomes
        the new root. Otherwise, creates a new tree.
        
        Args:
            new_state: New board state to search for
            new_player (int): Player whose turn it is now (1 for white, -1 for black)
        """
        # If no tree exists, create a new one
        if self.tree_root is None:
            self.tree_root = MCTSNode(new_state, new_player, 
                                     rollout_type=self.rollout_type, 
                                     rollout_simulations=self.rollout_simulations)
            return
        
        # Check if current root already matches
        if self._states_equal(self.tree_root.state, new_state):
            # Same state, just update player if needed
            if self.tree_root.player != new_player:
                self.tree_root.player = new_player
            return
        
        # Search for the new state in the tree (search up to 2 levels deep)
        # This covers: root -> child (opponent's move) -> grandchild (our next move)
        matching_node = self._find_node_by_state(self.tree_root, new_state, max_depth=2)
        
        if matching_node is not None:
            # Found a matching node - reuse that subtree
            # Disconnect from parent to make it the new root
            matching_node.parent = None
            matching_node.player = new_player  # Update player if needed
            self.tree_root = matching_node
        else:
            # Can't reuse - create new tree with rollout settings
            self.tree_root = MCTSNode(new_state, new_player,
                                    rollout_type=self.rollout_type,
                                    rollout_simulations=self.rollout_simulations)
    
    def get_move(self, board, valid_moves):
        if not valid_moves:
            return None
        
        # Convert board to list of lists (handle numpy arrays)
        if isinstance(board, np.ndarray):
            state_copy = board.tolist()
        else:
            state_copy = [list(row) for row in board]
        
        # Update tree root to current state (reuse tree if possible)
        self._update_tree_root(state_copy, self.player)
        
        # Run MCTS search starting from the (possibly reused) tree root
        best_move = mcts_search_with_tree(self.tree_root, iterations=self.iterations)
        
        # Store the current state for next time
        self.last_board_state = state_copy

        # If MCTS returns None or invalid move, fall back to random
        if best_move is None or best_move not in valid_moves:
            return random.choice(valid_moves)

        return best_move

class MinimaxAgent(BaseAgent):
    """
    Minimax agent with alpha-beta pruning and Othello-specific heuristics.
    
    This agent uses the minimax algorithm with optional alpha-beta pruning to
    search the game tree. It evaluates board positions using multiple Othello
    heuristics: corner control, edge control, mobility, stability, and disc count.
    Moves are ordered by heuristic value to improve alpha-beta pruning efficiency.
    
    Attributes:
        player (int): Player value (1 for white, -1 for black)
        abpruning (bool): Whether to use alpha-beta pruning
        depth (int): Search depth for minimax algorithm
        edge_positions (list): Pre-computed list of edge positions
        weights (dict): Heuristic weight configuration
    """
    
    def __init__(self, player, abpruning=True, depth=3):
        """
        Initialize the Minimax agent.
        
        Args:
            player (int): Player value (1 for white, -1 for black)
            abpruning (bool, optional): Enable alpha-beta pruning. Defaults to True.
            depth (int, optional): Search depth. Defaults to 3.
        """
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
        """
        Lazy load game functions when needed.
        
        Returns:
            dict: Dictionary of game function references
        """
        if self._game_funcs is None:
            self._game_funcs = _get_game_functions()
        return self._game_funcs
    
    def _quick_evaluate_move(self, board, move, player):
        """
        Quick evaluation of a single move for move ordering.
        
        Provides a fast heuristic to order moves for better alpha-beta pruning.
        Corners are highly valued (100), edges moderately (10), interior moves low (1).
        
        Args:
            board (numpy.ndarray): Current board state
            move (tuple): (row, col) move to evaluate
            player (int): Player making the move (not used in this simple heuristic)
        
        Returns:
            int: Heuristic value (100 for corners, 10 for edges, 1 for interior)
        """
        # Simple heuristic: prefer corners, then edges
        r, c = move
        if (r, c) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
            return 100  # Corners are best
        elif (r, c) in self.edge_positions:
            return 10   # Edges are good
        else:
            return 1    # Interior moves
    
    def _order_moves(self, board, moves, player):
        """
        Order moves by heuristic value to improve alpha-beta pruning.
        
        Sorts moves by quick evaluation score (best first) so that promising
        moves are explored early, allowing alpha-beta pruning to eliminate
        more branches.
        
        Args:
            board (numpy.ndarray): Current board state
            moves (list): List of (row, col) move tuples
            player (int): Player making the move
        
        Returns:
            list: Sorted list of moves (best first)
        """
        # Sort moves by quick evaluation (best first)
        return sorted(moves, key=lambda m: self._quick_evaluate_move(board, m, player), reverse=True)
    
    def evaluate_heuristic(self, board, player):
        """
        Evaluate board state using Othello-specific heuristics.
        
        Combines multiple strategic factors into a single score:
        - Corner control (weight: 25): Most important, unflippable positions
        - Stability (weight: 15): Stable discs that cannot be flipped
        - Mobility (weight: 10): Number of valid moves available
        - Edge control (weight: 5): Control of edge positions
        - Disc count (weight: 1, scaled): Raw disc difference, weighted by game progress
        
        The score is from the perspective of the given player (positive = favorable).
        
        Args:
            board (numpy.ndarray): Current board state
            player (int): Player to evaluate for (1 for white, -1 for black)
        
        Returns:
            float: Heuristic score (higher = better for the player)
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
        """
        Check if the game is over (no moves for either player).
        
        Args:
            board (numpy.ndarray): Current board state
        
        Returns:
            bool: True if game is over, False otherwise
        """
        game_funcs = self._load_game_functions()
        get_valid_moves = game_funcs['get_valid_moves']
        return (len(get_valid_moves(board, 1)) == 0 and 
                len(get_valid_moves(board, -1)) == 0)
    
    def minimax(self, board, depth, player, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning and move ordering.
        
        Recursively searches the game tree to find the optimal move. Uses alpha-beta
        pruning (if enabled) to eliminate branches that cannot improve the result.
        Moves are ordered by heuristic value to maximize pruning efficiency.
        
        Args:
            board (numpy.ndarray): Current board state
            depth (int): Remaining search depth
            player (int): Current player (1 for white, -1 for black)
            alpha (float): Best value for maximizing player (lower bound)
            beta (float): Best value for minimizing player (upper bound)
            maximizing_player (bool): True if we're maximizing for self.player
        
        Returns:
            tuple: (best_score, best_move) where best_score is the heuristic score
                   and best_move is the (row, col) tuple, or (score, None) at terminal nodes
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
        """
        Get the best move using minimax search.
        
        Runs minimax algorithm with alpha-beta pruning (if enabled) to find
        the optimal move according to the heuristic evaluation function.
        
        Args:
            board (numpy.ndarray): Current board state
            valid_moves (list): List of (row, col) tuples representing valid moves
        
        Returns:
            tuple: (row, col) tuple representing the best move
        """
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
    """
    Agent that selects moves randomly from valid moves.
    
    This agent provides a baseline for comparison with more sophisticated agents.
    It simply selects a random valid move each turn.
    """
    
    def get_move(self, board, valid_moves):
        """
        Select a random move from the valid moves.
        
        Args:
            board (numpy.ndarray): Current board state (not used)
            valid_moves (list): List of (row, col) tuples representing valid moves
        
        Returns:
            tuple: (row, col) tuple of a randomly selected move, or None if no moves
        """
        if not valid_moves:
            return None
        return random.choice(valid_moves)