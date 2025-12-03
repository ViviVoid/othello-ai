# Andy Dao (daoa@msoe.edu)
# Batch Job Script for Othello Game Simulations

import json
import argparse
import time
import os
from datetime import datetime
from main import run_game
import numpy as np
import tracemalloc


def run_batch_simulation(config_file, num_games=100, output_dir="results"):
    """
    Run multiple games and collect statistics.
    
    Args:
        config_file: Path to JSON configuration file
        num_games: Number of games to run
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_file, "r") as f:
        environment_data = json.load(f)
    
    # Statistics tracking
    stats = {
        'black_wins': 0,
        'white_wins': 0,
        'draws': 0,
        'total_moves': [],
        'white_scores': [],
        'black_scores': [],
        'move_times': {'black': [], 'white': []},
        'memory_usage_mb': [],
        'config': environment_data,
        'num_games': num_games,
        'timestamp': datetime.now().isoformat()
    }
    
    black_agent_type = environment_data["agents"][0][0]
    white_agent_type = environment_data["agents"][1][0]
    
    print(f"Running {num_games} games: {black_agent_type} (Black) vs {white_agent_type} (White)")
    print("=" * 60)
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()
    
    for game_num in range(1, num_games + 1):
        if game_num % 100 == 0:
            print(f"Progress: {game_num}/{num_games} games completed")
        
        game_start = time.time()
        result = run_game(environment_data, display=False)
        game_time = time.time() - game_start
        
        # Get memory usage after game (current and peak)
        current, peak = tracemalloc.get_traced_memory()
        # Store current memory usage for this game (convert bytes to MB)
        current_memory_mb = current / (1024 * 1024)
        stats['memory_usage_mb'].append(current_memory_mb)
        
        if result:
            # Update statistics
            if result['winner_player'] == -1:
                stats['black_wins'] += 1
            elif result['winner_player'] == 1:
                stats['white_wins'] += 1
            else:
                stats['draws'] += 1
            
            stats['total_moves'].append(result['total_moves'])
            stats['white_scores'].append(result['white_score'])
            stats['black_scores'].append(result['black_score'])
    
    total_time = time.time() - start_time
    
    # Get final memory statistics before stopping
    current_final, peak_final = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate summary statistics
    stats['summary'] = {
        'black_win_rate': stats['black_wins'] / num_games * 100,
        'white_win_rate': stats['white_wins'] / num_games * 100,
        'draw_rate': stats['draws'] / num_games * 100,
        'avg_total_moves': np.mean(stats['total_moves']) if stats['total_moves'] else 0,
        'avg_white_score': np.mean(stats['white_scores']) if stats['white_scores'] else 0,
        'avg_black_score': np.mean(stats['black_scores']) if stats['black_scores'] else 0,
        'avg_white_score_diff': np.mean([w - b for w, b in zip(stats['white_scores'], stats['black_scores'])]) if stats['white_scores'] else 0,
        'total_time_seconds': total_time,
        'avg_time_per_game': total_time / num_games,
        'avg_memory_usage_mb': np.mean(stats['memory_usage_mb']) if stats['memory_usage_mb'] else 0,
        'max_memory_usage_mb': np.max(stats['memory_usage_mb']) if stats['memory_usage_mb'] else 0,
        'min_memory_usage_mb': np.min(stats['memory_usage_mb']) if stats['memory_usage_mb'] else 0,
        'peak_memory_usage_mb': peak_final / (1024 * 1024) if peak_final else 0
    }
    
    # Save results
    output_file = os.path.join(output_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total games: {num_games}")
    print(f"Black wins: {stats['black_wins']} ({stats['summary']['black_win_rate']:.2f}%)")
    print(f"White wins: {stats['white_wins']} ({stats['summary']['white_win_rate']:.2f}%)")
    print(f"Draws: {stats['draws']} ({stats['summary']['draw_rate']:.2f}%)")
    print(f"\nAverage moves per game: {stats['summary']['avg_total_moves']:.2f}")
    print(f"Average white score: {stats['summary']['avg_white_score']:.2f}")
    print(f"Average black score: {stats['summary']['avg_black_score']:.2f}")
    print(f"Average score difference (white - black): {stats['summary']['avg_white_score_diff']:.2f}")
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Average time per game: {stats['summary']['avg_time_per_game']:.4f} seconds")
    print(f"\nMemory usage:")
    print(f"  Average per game: {stats['summary']['avg_memory_usage_mb']:.2f} MB")
    print(f"  Peak (overall): {stats['summary']['peak_memory_usage_mb']:.2f} MB")
    print(f"  Maximum per game: {stats['summary']['max_memory_usage_mb']:.2f} MB")
    print(f"  Minimum per game: {stats['summary']['min_memory_usage_mb']:.2f} MB")
    print(f"\nResults saved to: {output_file}")
    
    return stats


def run_comparison_matrix(config_files, num_games=100, output_dir="results"):
    """
    Run comparisons between different agent configurations.
    
    Args:
        config_files: List of configuration file paths
        num_games: Number of games per configuration
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_results = []
    
    for config_file in config_files:
        print(f"\n{'='*60}")
        print(f"Running comparison for: {config_file}")
        print(f"{'='*60}")
        stats = run_batch_simulation(config_file, num_games, output_dir)
        comparison_results.append({
            'config_file': config_file,
            'stats': stats
        })
    
    # Save comparison summary
    comparison_file = os.path.join(output_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(comparison_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nComparison results saved to: {comparison_file}")
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Batch Othello Game Simulations")
    parser.add_argument(
        "-f", "--filename",
        default="example-minimax.json",
        help="Game environment file"
    )
    parser.add_argument(
        "-n", "--num-games",
        type=int,
        default=100,
        help="Number of games to run"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "-c", "--compare",
        nargs="+",
        help="Run comparison between multiple config files"
    )
    args = parser.parse_args()
    
    if args.compare:
        run_comparison_matrix(args.compare, args.num_games, args.output_dir)
    else:
        run_batch_simulation(args.filename, args.num_games, args.output_dir)


if __name__ == "__main__":
    main()

