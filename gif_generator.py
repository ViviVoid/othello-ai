# Andy Dao (daoa@msoe.edu)
# Othello Replay GIF Generator

import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


# Game constants (matching main.py)
BOARD_SIZE = 8
CELL_SIZE = 80
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE
GREEN = (0, 128, 0)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
GRAY = (160, 160, 160)
HIGHLIGHT = (200, 200, 0)


def draw_board_pil(board, valid_moves=None, move_info=None):
    """Draw the board using PIL for GIF generation."""
    # Create image with extra space for status text
    img = Image.new('RGB', (WINDOW_SIZE, WINDOW_SIZE + 100), GREEN)
    draw = ImageDraw.Draw(img)
    
    # Draw grid
    for row in range(BOARD_SIZE + 1):
        y = row * CELL_SIZE
        draw.line([(0, y), (WINDOW_SIZE, y)], fill=BLACK_COLOR, width=2)
    for col in range(BOARD_SIZE + 1):
        x = col * CELL_SIZE
        draw.line([(x, 0), (x, WINDOW_SIZE)], fill=BLACK_COLOR, width=2)
    
    # Draw valid moves (highlight)
    if valid_moves:
        for r, c in valid_moves:
            center_x = c * CELL_SIZE + CELL_SIZE // 2
            center_y = r * CELL_SIZE + CELL_SIZE // 2
            draw.ellipse(
                [center_x - 8, center_y - 8, center_x + 8, center_y + 8],
                fill=HIGHLIGHT
            )
    
    # Draw pieces
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == 1:  # White
                center_x = col * CELL_SIZE + CELL_SIZE // 2
                center_y = row * CELL_SIZE + CELL_SIZE // 2
                draw.ellipse(
                    [center_x - 30, center_y - 30, center_x + 30, center_y + 30],
                    fill=WHITE_COLOR,
                    outline=BLACK_COLOR,
                    width=2
                )
            elif board[row][col] == -1:  # Black
                center_x = col * CELL_SIZE + CELL_SIZE // 2
                center_y = row * CELL_SIZE + CELL_SIZE // 2
                draw.ellipse(
                    [center_x - 30, center_y - 30, center_x + 30, center_y + 30],
                    fill=BLACK_COLOR,
                    outline=WHITE_COLOR,
                    width=2
                )
    
    # Draw status text
    if move_info:
        whites = move_info.get('white_score', 0)
        blacks = move_info.get('black_score', 0)
        move_num = move_info.get('move_number', 0)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
                small_font = ImageFont.truetype("arial.ttf", 18)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        
        status_text = f"Move {move_num} | White: {whites} Black: {blacks}"
        draw.text((10, WINDOW_SIZE + 10), status_text, fill=BLACK_COLOR, font=font)
        
        if move_info.get('game_over'):
            winner = move_info.get('winner', 'Unknown')
            winner_text = f"Game Over! Winner: {winner}"
            draw.text((10, WINDOW_SIZE + 40), winner_text, fill=BLACK_COLOR, font=font)
        elif move_info.get('move'):
            player = move_info.get('player', 'Unknown').capitalize()
            move_pos = move_info.get('move', ())
            move_text = f"{player} played: ({move_pos[0]}, {move_pos[1]})"
            draw.text((10, WINDOW_SIZE + 40), move_text, fill=BLACK_COLOR, font=small_font)
    
    return img


def generate_gif(replay_file, output_file, duration=500, loop=0):
    """
    Generate a GIF from a replay file.
    
    Args:
        replay_file: Path to replay JSON file
        output_file: Path to output GIF file
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    with open(replay_file, "r") as f:
        replay_data = json.load(f)
    
    moves = replay_data['moves']
    frames = []
    
    print(f"Generating GIF from {len(moves)} moves...")
    
    for i, move_data in enumerate(moves):
        # Determine which board to use
        if 'board_after' in move_data:
            board = np.array(move_data['board_after'])
        elif 'board' in move_data:
            board = np.array(move_data['board'])
        else:
            continue
        
        # Get valid moves
        valid_moves = move_data.get('valid_moves', [])
        
        # Create move info for status text
        move_info = {
            'move_number': move_data.get('move_number', i),
            'white_score': move_data.get('white_score_after') or move_data.get('white_score', 0),
            'black_score': move_data.get('black_score_after') or move_data.get('black_score', 0),
            'player': move_data.get('player'),
            'move': move_data.get('move'),
            'game_over': move_data.get('game_over', False),
            'winner': move_data.get('winner')
        }
        
        # Draw frame
        frame = draw_board_pil(board, valid_moves, move_info)
        frames.append(frame)
        
        # Add extra frames for important moves (corners, game over)
        if move_data.get('move'):
            r, c = move_data['move']
            is_corner = (r, c) in [(0, 0), (0, 7), (7, 0), (7, 7)]
            if is_corner:
                # Add extra frame for corner moves
                frames.append(frame)
        
        if move_data.get('game_over'):
            # Add extra frames at the end
            for _ in range(3):
                frames.append(frame)
    
    # Save as GIF
    if frames:
        print(f"Saving GIF with {len(frames)} frames...")
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop
        )
        print(f"GIF saved to: {output_file}")
    else:
        print("Error: No frames to save.")


def main():
    parser = argparse.ArgumentParser(description="Generate GIF from Othello Replay")
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Replay file (JSON format)"
    )
    parser.add_argument(
        "-o", "--output",
        default="",
        help="Output GIF file (default: replay file name with .gif extension)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=500,
        help="Frame duration in milliseconds (default: 500)"
    )
    parser.add_argument(
        "-l", "--loop",
        type=int,
        default=0,
        help="Number of loops (0 = infinite, default: 0)"
    )
    args = parser.parse_args()
    
    # Determine output file name
    if not args.output:
        base_name = os.path.splitext(args.file)[0]
        args.output = f"{base_name}.gif"
    
    try:
        generate_gif(args.file, args.output, args.duration, args.loop)
    except FileNotFoundError:
        print(f"Error: Replay file '{args.file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in replay file '{args.file}'.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

