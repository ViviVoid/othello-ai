# othello-ai

Agents are stored in `agent.py`

Game is ran using `main.py`

## Installation

Install dependencies from requirements:
```bash
pip install -r requirements.txt
```

## Running the Game

### Interactive Mode (with display)
```bash
python main.py -f game-environments/example-minimax.json
```

### Headless Mode (no display, for batch processing)
```bash
python main.py -f game-environments/example-minimax.json --headless
```

### With Output File
```bash
python main.py -f game-environments/example-minimax.json -o game_result.json
```

### With Replay File
```bash
# Save replay file
python main.py -f game-environments/example-minimax.json --replay game_replay.json

# Save replay and generate GIF
python main.py -f game-environments/example-minimax.json --replay game_replay.json --gif
```

## Batch Simulations

Run multiple games for performance benchmarking:

```bash
# Run 100 games with default config
python batch_run.py -f game-environments/random-random.json -n 100

# Run with custom output directory
python batch_run.py -f game-environments/random-random.json -n 1000 -o results/

# Compare multiple configurations
python batch_run.py -c game-environments/random-random.json game-environments/mcts-random.json -n 100
```

## Configuration Files

Configuration files are JSON files that specify:
- `agents`: List of two agent configurations (black, white)
- `display`: Whether to show pygame display (optional)
- `num_games`: Number of games to run (optional, for batch scripts)

Agent types:
- `["human"]` - Human player (requires display)
- `["random"]` - Random move agent
- `["minimax", use_alpha_beta, depth]` - Minimax agent
- `["mcts"]` - Monte Carlo Tree Search agent

Example:
```json
{
  "agents": [
    ["minimax", true, 3],
    ["random"]
  ],
  "display": false,
  "num_games": 100
}
```

## Board State Heuristics

The game includes board state evaluation functions:
- `count_discs(board)` - Count white and black discs
- `get_mobility(board, player)` - Get number of valid moves for a player
- `count_stable_discs(board, player)` - Count stable (unflippable) discs
- `evaluate_board_state(board, player)` - Comprehensive board evaluation

## Replay Functionality

The game supports saving and replaying games with step-by-step navigation and undo/redo functionality.

### Saving Replays

Save a replay file during gameplay:
```bash
python main.py -f game-environments/example-minimax.json --replay game_replay.json
```

This creates a JSON file containing all moves, board states, and game information.

### Viewing Replays

Use the replay viewer to step through a saved game:
```bash
python replay_viewer.py -f game_replay.json
```

**Controls:**
- `←` / `→` : Step backward/forward through moves
- `Home` / `End` : Jump to start/end of game
- `U` / `R` : Undo/Redo navigation actions
- `Q` / `ESC` : Quit viewer

### Generating GIFs

Generate an animated GIF from a replay file:
```bash
# Basic GIF generation
python gif_generator.py -f game_replay.json

# Custom duration and output file
python gif_generator.py -f game_replay.json -o game_animation.gif -d 300

# Options:
# -d, --duration: Frame duration in milliseconds (default: 500)
# -l, --loop: Number of loops, 0 = infinite (default: 0)
```

Or automatically generate GIF after a game:
```bash
python main.py -f game-environments/example-minimax.json --replay game_replay.json --gif
```

## Project Structure

- `main.py` - Main game logic, pygame visualization, and game loop
- `agent.py` - Agent implementations (Human, Random, Minimax, MCTS)
- `batch_run.py` - Batch simulation script for performance analysis
- `replay_viewer.py` - Interactive replay viewer with undo/redo
- `gif_generator.py` - Generate animated GIFs from replay files
- `game-environments/` - Directory containing game configuration JSON files