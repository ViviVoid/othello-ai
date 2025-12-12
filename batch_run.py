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
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


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
        'move_times_by_position': [],  # All move times with position info
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
    
    # Create progress bar
    game_range = range(1, num_games + 1)
    if HAS_TQDM:
        pbar = tqdm(game_range, desc="Games", unit="game", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = game_range
    
    for game_num in pbar:
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
            
            # Collect per-move timing data
            if 'move_times' in result and result['move_times']:
                for move_data in result['move_times']:
                    player = move_data['player']
                    move_time = move_data['time_seconds']
                    stats['move_times'][player].append(move_time)
                    # Store with position information for progression analysis
                    stats['move_times_by_position'].append({
                        'move_number': move_data['move_number'],
                        'player': player,
                        'time_seconds': move_time,
                        'total_discs': move_data['total_discs'],
                        'valid_moves_count': move_data.get('valid_moves_count', 0)
                    })
        
        # Update progress bar description with current stats
        if HAS_TQDM:
            wins_so_far = stats['black_wins'] + stats['white_wins'] + stats['draws']
            if wins_so_far > 0:
                black_pct = (stats['black_wins'] / wins_so_far) * 100
                white_pct = (stats['white_wins'] / wins_so_far) * 100
                pbar.set_postfix({
                    'Black': f'{stats["black_wins"]} ({black_pct:.1f}%)',
                    'White': f'{stats["white_wins"]} ({white_pct:.1f}%)',
                    'Draws': stats['draws']
                })
    
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
    
    # Calculate move timing statistics
    if stats['move_times']['black']:
        stats['summary']['avg_move_time_black'] = np.mean(stats['move_times']['black'])
        stats['summary']['max_move_time_black'] = np.max(stats['move_times']['black'])
        stats['summary']['min_move_time_black'] = np.min(stats['move_times']['black'])
    else:
        stats['summary']['avg_move_time_black'] = 0
        stats['summary']['max_move_time_black'] = 0
        stats['summary']['min_move_time_black'] = 0
    
    if stats['move_times']['white']:
        stats['summary']['avg_move_time_white'] = np.mean(stats['move_times']['white'])
        stats['summary']['max_move_time_white'] = np.max(stats['move_times']['white'])
        stats['summary']['min_move_time_white'] = np.min(stats['move_times']['white'])
    else:
        stats['summary']['avg_move_time_white'] = 0
        stats['summary']['max_move_time_white'] = 0
        stats['summary']['min_move_time_white'] = 0
    
    # Analyze timing progression over game course
    if stats['move_times_by_position']:
        # Group by move number (early/mid/late game)
        move_numbers = [m['move_number'] for m in stats['move_times_by_position']]
        max_move = max(move_numbers) if move_numbers else 0
        
        # Divide game into thirds
        early_threshold = max_move // 3
        mid_threshold = 2 * max_move // 3
        
        early_times = [m['time_seconds'] for m in stats['move_times_by_position'] 
                      if m['move_number'] <= early_threshold]
        mid_times = [m['time_seconds'] for m in stats['move_times_by_position'] 
                     if early_threshold < m['move_number'] <= mid_threshold]
        late_times = [m['time_seconds'] for m in stats['move_times_by_position'] 
                     if m['move_number'] > mid_threshold]
        
        stats['summary']['timing_by_game_phase'] = {
            'early_game': {
                'avg_time': np.mean(early_times) if early_times else 0,
                'max_time': np.max(early_times) if early_times else 0,
                'min_time': np.min(early_times) if early_times else 0,
                'move_count': len(early_times)
            },
            'mid_game': {
                'avg_time': np.mean(mid_times) if mid_times else 0,
                'max_time': np.max(mid_times) if mid_times else 0,
                'min_time': np.min(mid_times) if mid_times else 0,
                'move_count': len(mid_times)
            },
            'late_game': {
                'avg_time': np.mean(late_times) if late_times else 0,
                'max_time': np.max(late_times) if late_times else 0,
                'min_time': np.min(late_times) if late_times else 0,
                'move_count': len(late_times)
            }
        }
        
        # Analyze by total discs (board fill progression)
        disc_ranges = {
            'early': (0, 20),   # 0-20 discs
            'mid': (20, 45),    # 21-45 discs
            'late': (45, 64)    # 46-64 discs
        }
        
        timing_by_discs = {}
        for phase, (min_discs, max_discs) in disc_ranges.items():
            phase_times = [m['time_seconds'] for m in stats['move_times_by_position'] 
                          if min_discs <= m['total_discs'] < max_discs]
            timing_by_discs[phase] = {
                'avg_time': np.mean(phase_times) if phase_times else 0,
                'max_time': np.max(phase_times) if phase_times else 0,
                'min_time': np.min(phase_times) if phase_times else 0,
                'move_count': len(phase_times)
            }
        
        stats['summary']['timing_by_board_fill'] = timing_by_discs
    else:
        stats['summary']['timing_by_game_phase'] = {}
        stats['summary']['timing_by_board_fill'] = {}
    
    # Save results (convert numpy types to native Python types for JSON serialization)
    output_file = os.path.join(output_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(convert_numpy_types(stats), f, indent=2)
    
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
    
    # Print move timing statistics
    if stats['move_times']['black'] or stats['move_times']['white']:
        print(f"\nMove Computation Time:")
        if stats['move_times']['black']:
            print(f"  Black ({black_agent_type}):")
            print(f"    Average: {stats['summary']['avg_move_time_black']:.4f} seconds")
            print(f"    Max: {stats['summary']['max_move_time_black']:.4f} seconds")
            print(f"    Min: {stats['summary']['min_move_time_black']:.4f} seconds")
        if stats['move_times']['white']:
            print(f"  White ({white_agent_type}):")
            print(f"    Average: {stats['summary']['avg_move_time_white']:.4f} seconds")
            print(f"    Max: {stats['summary']['max_move_time_white']:.4f} seconds")
            print(f"    Min: {stats['summary']['min_move_time_white']:.4f} seconds")
        
        # Print timing by game phase
        if 'timing_by_game_phase' in stats['summary'] and stats['summary']['timing_by_game_phase']:
            print(f"\n  Timing by Game Phase:")
            for phase, data in stats['summary']['timing_by_game_phase'].items():
                if data['move_count'] > 0:
                    print(f"    {phase.capitalize()}: avg={data['avg_time']:.4f}s, "
                          f"max={data['max_time']:.4f}s, moves={data['move_count']}")
        
        # Print timing by board fill
        if 'timing_by_board_fill' in stats['summary'] and stats['summary']['timing_by_board_fill']:
            print(f"\n  Timing by Board Fill:")
            for phase, data in stats['summary']['timing_by_board_fill'].items():
                if data['move_count'] > 0:
                    print(f"    {phase.capitalize()} (discs): avg={data['avg_time']:.4f}s, "
                          f"max={data['max_time']:.4f}s, moves={data['move_count']}")
    
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
        default="game-environments/example-minimax.json",
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

