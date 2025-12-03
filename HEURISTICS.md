# Othello AI Heuristics Overview

This document provides a high-level overview of the heuristics used in the Othello AI implementation for board evaluation and move selection.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Heuristics](#core-heuristics)
3. [Heuristic Weights](#heuristic-weights)
4. [Move Ordering](#move-ordering)
5. [Implementation Details](#implementation-details)

## Introduction

The Othello AI uses a combination of strategic heuristics to evaluate board positions and make optimal moves. These heuristics are based on established Othello strategy principles and are weighted to reflect their relative importance in different game phases.

The primary evaluation function combines multiple factors into a single score that represents the desirability of a board position from the current player's perspective. Higher scores indicate more favorable positions.

## Core Heuristics

### 1. Corner Control

**Weight: 25** (Highest priority)

Corners are the most valuable positions in Othello because once captured, they cannot be flipped by the opponent. Controlling corners provides a stable foundation and often leads to controlling adjacent edges.

- **Calculation**: Counts the number of corners (positions (0,0), (0,7), (7,0), (7,7)) controlled by each player
- **Score**: `weight × (player_corners - opponent_corners)`
- **Rationale**: Corner control is fundamental to Othello strategy and often determines game outcomes

### 2. Stability

**Weight: 15** (High priority)

Stable discs are pieces that cannot be flipped by any legal sequence of moves. These provide a secure advantage that the opponent cannot easily reverse.

- **Calculation**: 
  - Counts corner discs (always stable)
  - Counts edge discs adjacent to stable corners
  - Only computed after 20 discs are on the board (optimization)
- **Score**: `weight × (player_stable_discs - opponent_stable_discs)`
- **Rationale**: Stable discs represent permanent advantages and are crucial in endgame scenarios

### 3. Mobility

**Weight: 10** (Moderate-high priority)

Mobility measures the number of legal moves available to a player. Having more move options provides flexibility and can force the opponent into disadvantageous positions.

- **Calculation**: Counts the number of valid moves available to each player
- **Score**: `weight × (player_mobility - opponent_mobility)`
- **Rationale**: High mobility allows a player to control the game flow and avoid being forced into bad moves

### 4. Edge Control

**Weight: 5** (Moderate priority)

Edge positions (excluding corners) are valuable because they can lead to corner control and provide stability. However, they must be used carefully as they can be vulnerable to corner capture.

- **Calculation**: Counts edge positions (all positions on the perimeter except corners) controlled by each player
- **Score**: `weight × (player_edges - opponent_edges)`
- **Rationale**: Edge control is important but less critical than corners; it's a means to an end rather than an end in itself

### 5. Disc Count

**Weight: 1** (Base weight, scales with game progress)

The raw count of discs on the board. This becomes more important as the game progresses toward the endgame.

- **Calculation**: 
  - Computes disc difference: `player_discs - opponent_discs`
  - Applies dynamic weighting based on game progress: `base_weight × (total_discs / 64)`
- **Score**: `endgame_weight × disc_difference`
- **Rationale**: 
  - Early game: Disc count is less important than positional advantages
  - Endgame: Disc count becomes the primary factor as the board fills up
  - The dynamic weighting ensures the heuristic adapts to game phase

## Heuristic Weights

The heuristics are combined using weighted linear combination:

```python
total_score = corner_score + edge_score + mobility_score + stability_score + disc_score
```

### Weight Configuration

| Heuristic | Weight | Rationale |
|-----------|--------|-----------|
| Corner | 25 | Highest priority - corners are unflippable |
| Stability | 15 | High priority - stable discs provide permanent advantage |
| Mobility | 10 | Moderate-high - flexibility and control |
| Edge | 5 | Moderate - valuable but secondary to corners |
| Disc Count | 1 (scaled) | Base weight, increases with game progress |

### Dynamic Weighting

- **Stability**: Only computed after 20 discs are placed (early game optimization)
- **Disc Count**: Weight scales linearly with board fill: `weight × (total_discs / 64)`
  - At start (4 discs): ~0.06× weight
  - Mid-game (32 discs): ~0.5× weight  
  - Endgame (60 discs): ~0.94× weight

## Move Ordering

To improve alpha-beta pruning efficiency, moves are ordered by a quick evaluation before the full minimax search:

### Quick Evaluation Function

1. **Corners**: Score = 100 (highest priority)
2. **Edges**: Score = 10 (moderate priority)
3. **Interior**: Score = 1 (lowest priority)

This ordering ensures that the most promising moves are explored first, allowing alpha-beta pruning to eliminate more branches early in the search.

## Implementation Details

### Evaluation Function

The main evaluation function (`evaluate_heuristic` in `MinimaxAgent`) combines all heuristics:

1. **Corner Control**: Direct count of corner positions
2. **Edge Control**: Uses pre-computed edge position list for efficiency
3. **Mobility**: Calculates valid moves for both players
4. **Stability**: Computed only when `total_discs > 20` (performance optimization)
5. **Disc Count**: Dynamically weighted based on game progress

### Performance Optimizations

- **Pre-computed edge positions**: Edge positions are calculated once during agent initialization
- **Conditional stability calculation**: Stability is only computed after 20 discs are placed
- **Move ordering**: Moves are sorted by quick evaluation to improve alpha-beta pruning
- **Lazy function loading**: Game functions are loaded on-demand to avoid circular imports

### Board State Evaluation

The `evaluate_board_state` function in `main.py` provides a comprehensive breakdown of board metrics:

- Disc counts (player and opponent)
- Disc difference
- Mobility (player and opponent)
- Mobility difference
- Stable discs (player and opponent)
- Stable discs difference
- Total discs on board

This function is useful for analysis and debugging, providing detailed metrics beyond the single heuristic score.

## Strategic Principles

The heuristic design follows these Othello strategic principles:

1. **Position over Material**: Early game focuses on positional advantages (corners, stability) rather than disc count
2. **Stability Matters**: Permanent advantages (stable discs) are highly valued
3. **Flexibility is Key**: Mobility provides options and control
4. **Endgame Transition**: As the board fills, disc count becomes increasingly important
5. **Corner Dominance**: Corner control is the single most important factor

## Future Enhancements

Potential improvements to the heuristic system:

- **Parity**: Track whether the number of empty squares favors the current player
- **Frontier Discs**: Penalize discs adjacent to empty squares (vulnerable positions)
- **X-Squares and C-Squares**: Special handling for positions adjacent to corners
- **Pattern Recognition**: Recognize common Othello patterns and positions
- **Adaptive Weights**: Adjust weights based on game phase or opponent behavior

---

*This heuristic system is used by the Minimax agent with alpha-beta pruning for optimal move selection in Othello games.*

