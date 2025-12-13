# Othello AI

An implementation of the Othello (Reversi) board game with multiple AI agents, including Minimax with alpha-beta pruning and Monte Carlo Tree Search (MCTS), featuring game visualization, replay functionality, and comprehensive performance analysis tools.

## Project Contributors

- **Andy Dao** (daoa@msoe.edu)
- **Sierra Andrews** (andrewss@msoe.edu)

## Project Overview

This project implements a complete Othello game system with multiple AI agent types for playing and analyzing games. The system includes:

- **Game Engine**: Core Othello game logic with board state management and move validation
- **AI Agents**: Multiple agent implementations including Human, Random, Minimax (with alpha-beta pruning), and MCTS
- **Visualization**: Pygame-based graphical interface for interactive gameplay
- **Replay System**: Save and replay games with step-by-step navigation and GIF generation
- **Performance Analysis**: Batch simulation tools for comparing agent performance and analyzing computational complexity
- **Heuristic Evaluation**: Comprehensive board state evaluation using multiple strategic heuristics (corner control, mobility, stability, edge control, disc count)

The project enables detailed study of AI agent performance, computational complexity analysis, and strategic game playing through configurable agent matchups and batch simulation capabilities.

## File and Folder Descriptions

### Core Game Files

- **`main.py`** - Main game engine with pygame visualization, game loop, board state heuristics, and game configuration loading. Handles game execution, move processing, and replay data generation.

- **`agent.py`** - AI agent implementations including HumanAgent (mouse input), RandomAgent (random moves), MinimaxAgent (minimax with alpha-beta pruning and heuristic evaluation), RandomMCTSAgent (Monte Carlo Tree Search with tree reuse), and MCTSNode class for tree structure management.

### Batch Processing and Analysis

- **`batch_run.py`** - Batch simulation script for running multiple games and collecting statistical data including win rates, move times, memory usage, and game progression metrics. Generates JSON result files with comprehensive performance data.

- **`generate_batch_replays.py`** - Script to generate replay files and GIFs from batch simulation results. Re-runs games with replay saving enabled and optionally generates animated GIFs for visualization.

### Replay and Visualization

- **`replay_viewer.py`** - Interactive replay viewer with step-by-step navigation (forward/backward), jump to start/end, undo/redo functionality, and pygame-based board visualization showing move history and game state.

- **`gif_generator.py`** - Generates animated GIF files from replay JSON files using PIL/Pillow, creating frame-by-frame animations of game progression with move information and board states.

### Configuration and Documentation

- **`game-environments/`** - Directory containing JSON configuration files that specify agent matchups, agent parameters (depth, iterations, rollout types), and game settings. Includes subdirectory `mcts-testing-environments/` with hyperparameter test configurations.

- **`requirements.txt`** - Python package dependencies including numpy, pygame, Pillow, and tqdm with version specifications.

- **`HEURISTICS.md`** - Detailed documentation of board evaluation heuristics including corner control, mobility, stability, edge control, and disc count with weight explanations and strategic rationale.

- **`LICENSE.md`** - Project license information.

### Output Directories

- **`results/`** - Directory containing batch simulation results as JSON files with statistics including win rates, move times, memory usage, and game progression analysis. Includes subdirectory `mcts-data/` for MCTS hyperparameter test results.

- **`replays/`** - Directory containing saved game replay files (JSON format) and generated GIF animations organized by batch run or individual games.

- **`__pycache__/`** - Python bytecode cache directory (automatically generated, not version controlled).

### Shell Scripts

- **`run_heuristic_tests.sh`** - Bash script to run comprehensive heuristic testing suite comparing MCTS, Minimax, and Random agents in various matchups with colored output and summary reporting.

- **`run_mcts_minimax_tests.sh`** - Bash script for systematic MCTS hyperparameter testing across different iteration counts and rollout configurations against Minimax, generating organized test results with progress tracking.

- **`generate_all_batch_replays.sh`** - Bash script to batch process all JSON result files in results/mcts-data directory, generating replay files and optional GIFs for all batch simulations.

### Other Files

- **`mcts-analysis.ipynb`** - Jupyter notebook for analyzing MCTS performance data and results.

- **`project-proposal.pdf`** - Original project proposal document.

- **`*.log`** - Log files from test runs containing execution logs and timing information.

## Compilation and Running Instructions

### Prerequisites

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- `numpy` (2.3.4) - Numerical computations for board state representation
- `pygame` (2.6.1) - Game visualization and user input handling
- `Pillow` (>=10.0.0) - Image processing for GIF generation
- `tqdm` (>=4.66.0) - Progress bars for batch operations (optional but recommended)

### Running the Game

#### Interactive Mode (with display)
```bash
python main.py -f game-environments/minimax-random.json
```

#### Headless Mode (no display, for batch processing)
```bash
python main.py -f game-environments/minimax-random.json --headless
```

#### With Output File
Save game results to a JSON file:
```bash
python main.py -f game-environments/minimax-random.json -o game_result.json
```

#### With Replay File
Save replay data for later viewing:
```bash
python main.py -f game-environments/minimax-random.json --replay game_replay.json
```

#### Save Replay and Generate GIF
Automatically generate GIF after game:
```bash
python main.py -f game-environments/minimax-random.json --replay game_replay.json --gif
```

### Command Line Arguments

- `-f, --filename`: Path to game environment configuration file (required)
- `-o, --outputfile`: Path to save game result JSON file (optional)
- `--headless`: Run without pygame display (optional, for batch processing)
- `--replay`: Path to save replay JSON file (optional)
- `--gif`: Generate GIF from replay file after game (requires --replay)

### Batch Simulations

Run multiple games for performance benchmarking:
```bash
# Run 100 games with default config
python batch_run.py -f game-environments/random-random.json -n 100

# Run with custom output directory
python batch_run.py -f game-environments/random-random.json -n 1000 -o results/

# Compare multiple configurations
python batch_run.py -c game-environments/random-random.json game-environments/mcts-random.json -n 100
```

Batch run arguments:
- `-f, --filename`: Configuration file for single batch run
- `-n, --num-games`: Number of games to run (default: 100)
- `-o, --output-dir`: Output directory for results (default: "results")
- `-c, --compare`: List of config files for comparison matrix

### Replay Viewer

View saved replays interactively:
```bash
python replay_viewer.py -f game_replay.json
```

**Controls:**
- `←` / `→` : Step backward/forward through moves
- `Home` / `End` : Jump to start/end of game
- `U` / `R` : Undo/Redo navigation actions
- `Q` / `ESC` : Quit viewer

### GIF Generation

Generate animated GIF from replay file:
```bash
# Basic GIF generation
python gif_generator.py -f game_replay.json

# Custom duration and output file
python gif_generator.py -f game_replay.json -o game_animation.gif -d 300

# Options:
# -d, --duration: Frame duration in milliseconds (default: 500)
# -l, --loop: Number of loops, 0 = infinite (default: 0)
```

### Shell Scripts

I made these for running in linux. If you can't run these with WSL or want to run these in Windows, chatgpt them into powershell scripts.

#### Running Heuristic Tests
```bash
chmod +x run_heuristic_tests.sh
./run_heuristic_tests.sh
# Or with custom number of games:
NUM_GAMES=50 ./run_heuristic_tests.sh
```

#### Running MCTS vs Minimax Tests
```bash
chmod +x run_mcts_minimax_tests.sh
./run_mcts_minimax_tests.sh
# Or with custom number of games:
NUM_GAMES=50 ./run_mcts_minimax_tests.sh
```

#### Generating Batch Replays
```bash
chmod +x generate_all_batch_replays.sh
./generate_all_batch_replays.sh
# Or without GIF generation:
GENERATE_GIFS=false ./generate_all_batch_replays.sh
```

## Configuration Files

Configuration files are JSON files that specify agent types and parameters:

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

### Agent Types

- **`["human"]`** - Human player (requires display mode)
- **`["random"]`** - Random move agent
- **`["minimax", use_alpha_beta, depth]`** - Minimax agent with optional alpha-beta pruning and search depth
- **`["mcts", iterations, rollout_type, rollout_simulations]`** - MCTS agent with iteration count, rollout strategy ('random' or 'minimax'), and number of rollout simulations

Example configurations are provided in the `game-environments/` directory.

## Results and Outputs

### Batch Simulation Results

Batch simulation results are saved in the `results/` directory as JSON files named `batch_results_YYYYMMDD_HHMMSS.json`. Each file contains:

- **Win Statistics**: Black wins, white wins, draws, and win rates
- **Score Statistics**: Average scores, score differences
- **Move Statistics**: Total moves, move counts per game
- **Timing Data**: 
  - Per-player average, min, max move computation times
  - Timing by game phase (early/mid/late game)
  - Timing by board fill progression
- **Memory Usage**: Average, peak, min, max memory consumption per game
- **Configuration**: Original game configuration used for the batch
- **Timestamp**: When the batch was run

### Interpreting Results

1. **Win Rates**: Compare win percentages to evaluate agent strength
2. **Move Times**: Analyze computational complexity (higher times indicate more complex algorithms or deeper searches)
3. **Timing by Phase**: Understand when agents spend more computation time (early game has fewer constraints, late game has more strategic depth)
4. **Memory Usage**: Monitor resource consumption for scalability analysis
5. **Score Differences**: Measure margin of victory to assess dominance

### Replay Files

Replay files are JSON files containing:
- Complete move history with board states before and after each move
- Player information and move coordinates
- Valid moves at each step
- Final game result and winner information
- Configuration used for the game
- Timestamp of when the game was played

### GIF Files

Generated GIF files show animated progression of the game with:
- Frame-by-frame board states
- Move information and player actions
- Score display
- Game over information

GIFs are saved alongside replay JSON files in the `replays/` directory, organized by batch run or individual game names.

## Shell Scripts

### `run_heuristic_tests.sh`

**Purpose**: Runs a comprehensive test suite comparing different agent matchups (MCTS vs Random, Minimax vs Random, MCTS vs Minimax, etc.) for heuristic evaluation and complexity analysis.

**Features**:
- Runs multiple agent matchup configurations
- Colored output for progress tracking
- Detailed logging to timestamped log files
- Summary report generation
- Error handling with continuation on failure

**Usage**:
```bash
./run_heuristic_tests.sh
NUM_GAMES=200 ./run_heuristic_tests.sh  # Custom number of games
```

**Output**: Results saved to `results/` directory, log file `test_run_YYYYMMDD_HHMMSS.log`

### `run_mcts_minimax_tests.sh`

**Purpose**: Systematically tests MCTS hyperparameter combinations (iterations × rollouts) against Minimax agents, testing both player order configurations (MCTS first and Minimax first).

**Features**:
- Tests 3 iteration levels (100, 1000, 10000) × 3 rollout levels (1, 5, 10) × 2 player orders = 18 configurations
- Organized output by iteration level
- Comprehensive summary report with win rates and timing statistics
- Progress tracking with test counters

**Usage**:
```bash
./run_mcts_minimax_tests.sh
NUM_GAMES=100 ./run_mcts_minimax_tests.sh  # Custom number of games
```

**Output**: Results saved to `results/mcts-data/` directory, log file `mcts_test_run_YYYYMMDD_HHMMSS.log`

### `generate_all_batch_replays.sh`

**Purpose**: Batch processes all JSON result files in `results/mcts-data/` directory to generate replay files and optional GIF animations.

**Features**:
- Finds all JSON files in results directory
- Generates replays for all games in each batch result
- Optional GIF generation (can be disabled)
- Progress tracking and error handling

**Usage**:
```bash
./generate_all_batch_replays.sh
GENERATE_GIFS=false ./generate_all_batch_replays.sh  # Skip GIF generation
```

**Output**: Replay JSON files and GIFs saved to `replays/{batch_file_name}/` directories

**Note**: All shell scripts require executable permissions (`chmod +x script_name.sh`) and assume Python 3 is available as `python3`.
