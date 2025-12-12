#!/usr/bin/env python3
"""
Batch Replay and GIF Generator from Batch Simulation Results

This module generates replay files and animated GIFs from batch simulation
result files. It re-runs games with replay saving enabled to create detailed
replay data and optionally generates GIF animations for visualization.

The module enables:
- Converting batch result files into individual game replays
- Generating animated GIFs showing game progression
- Organizing replays by batch run in dedicated directories
- Batch processing of multiple result files

Authors:
    Andy Dao (daoa@msoe.edu)
"""

# Andy Dao (daoa@msoe.edu)
# Generate replays and GIFs from batch results

import json
import argparse
import os
import sys
from pathlib import Path
from main import run_game
from gif_generator import generate_gif

def generate_replays_from_batch(batch_results_file, output_dir=None, generate_gifs=True):
    """
    Generate replay files and GIFs from a batch results JSON file.
    
    Loads a batch results file, extracts the configuration, and re-runs each
    game with replay saving enabled. Optionally generates animated GIF files
    for each game. Replays are saved in a directory structure based on the
    batch file name.
    
    Args:
        batch_results_file (str): Path to batch results JSON file
        output_dir (str, optional): Directory to save replays. If None, uses
                                   "replays/{batch_file_name}". Defaults to None.
        generate_gifs (bool, optional): Whether to generate GIF files. Defaults to True.
    """
    # Load batch results
    with open(batch_results_file, 'r') as f:
        batch_data = json.load(f)
    
    # Extract configuration
    config = batch_data.get('config', {})
    num_games = batch_data.get('num_games', 100)
    total_moves_list = batch_data.get('total_moves', [])
    
    # Create output directory
    if output_dir is None:
        # Create directory based on batch file name
        batch_name = Path(batch_results_file).stem
        output_dir = f"replays/{batch_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating replays for {num_games} games from {batch_results_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Re-run games with replay saving enabled
    for game_num in range(1, num_games + 1):
        replay_file = os.path.join(output_dir, f"game_{game_num:04d}.json")
        
        # Run game with replay saving
        result = run_game(
            config,
            display=False,
            save_replay=True,
            replay_file=replay_file
        )
        
        if result and result.get('replay_data'):
            # Generate GIF if requested
            if generate_gifs:
                gif_file = os.path.join(output_dir, f"game_{game_num:04d}.gif")
                try:
                    generate_gif(replay_file, gif_file, duration=500, loop=0)
                    print(f"✓ Game {game_num}/{num_games}: {replay_file} -> {gif_file}")
                except Exception as e:
                    print(f"✗ Game {game_num}/{num_games}: Failed to generate GIF: {e}")
            else:
                print(f"✓ Game {game_num}/{num_games}: {replay_file}")
        else:
            print(f"✗ Game {game_num}/{num_games}: Failed to generate replay")
    
    print("=" * 60)
    print(f"Replays saved to: {output_dir}")
    if generate_gifs:
        print(f"GIFs saved to: {output_dir}")


def main():
    """
    Main entry point for batch replay generation script.
    
    Parses command-line arguments and generates replays and GIFs from a
    batch results file.
    
    Command-line arguments:
        -f, --file: Batch results JSON file (required)
        -o, --output-dir: Output directory for replays (optional)
        --no-gifs: Skip GIF generation, only create JSON replays
    """
    parser = argparse.ArgumentParser(
        description="Generate replays and GIFs from batch results"
    )
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Batch results JSON file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for replays (default: replays/{batch_file_name})"
    )
    parser.add_argument(
        "--no-gifs",
        action="store_true",
        help="Don't generate GIF files, only JSON replays"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    generate_replays_from_batch(
        args.file,
        output_dir=args.output_dir,
        generate_gifs=not args.no_gifs
    )


if __name__ == "__main__":
    main()

