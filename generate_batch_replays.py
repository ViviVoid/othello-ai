#!/usr/bin/env python3
# Andy Dao (daoa@msoe.edu)
# Sierra Andrews (andrewss@msoe.edu)
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
    
    Args:
        batch_results_file: Path to batch results JSON file
        output_dir: Directory to save replays (default: based on batch file name)
        generate_gifs: Whether to generate GIF files
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

