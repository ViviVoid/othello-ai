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
python main.py -f example-minimax.json
```

### Headless Mode (no display, for batch processing)
```bash
python main.py -f example-minimax.json --headless
```

### With Output File
```bash
python main.py -f example-minimax.json -o game_result.json
```

## Batch Simulations

Run multiple games for performance benchmarking:

```bash
# Run 100 games with default config
python batch_run.py -f example-random-vs-random.json -n 100

# Run with custom output directory
python batch_run.py -f example-random-vs-random.json -n 1000 -o results/

# Compare multiple configurations
python batch_run.py -c example-random-vs-random.json example-mcts-vs-random.json -n 100
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

## Project Structure

- `main.py` - Main game logic, pygame visualization, and game loop
- `agent.py` - Agent implementations (Human, Random, Minimax, MCTS)
- `batch_run.py` - Batch simulation script for performance analysis
- `example-*.json` - Example configuration files