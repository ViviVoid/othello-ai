# Andy Dao (daoa@msoe.edu)
# Othello Replay Viewer with Undo/Redo

import pygame
import argparse
import json
import sys
import numpy as np
from main import (
    BOARD_SIZE, CELL_SIZE, WINDOW_SIZE,
    GREEN, BLACK, WHITE, GRAY, HIGHLIGHT,
    init_pygame, draw_board, count_discs
)


class ReplayViewer:
    def __init__(self, replay_file):
        """Initialize the replay viewer with a replay file."""
        with open(replay_file, "r") as f:
            self.replay_data = json.load(f)
        
        self.moves = self.replay_data['moves']
        self.current_move_index = 0
        self.history = []  # For undo/redo
        self.future = []   # For redo
        
        # Initialize pygame
        init_pygame()
        self.screen = pygame.display.get_surface()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)
        
        # Get initial board state
        if self.moves:
            self.current_board = np.array(self.moves[0]['board'])
        else:
            self.current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    
    def get_current_state(self):
        """Get the current board state and move info."""
        if self.current_move_index < len(self.moves):
            move_data = self.moves[self.current_move_index]
            # Prefer board_after if available (shows state after move)
            if 'board_after' in move_data:
                return np.array(move_data['board_after']), move_data
            elif 'board' in move_data:
                return np.array(move_data['board']), move_data
        
        # Fallback: if at the end, use the last move's board_after
        if self.current_move_index > 0:
            prev_move = self.moves[self.current_move_index - 1]
            if 'board_after' in prev_move:
                return np.array(prev_move['board_after']), prev_move
            elif 'board' in prev_move:
                return np.array(prev_move['board']), prev_move
        
        return self.current_board, None
    
    def step_forward(self):
        """Move forward one step in the replay."""
        if self.current_move_index < len(self.moves) - 1:
            # Save current state to history for undo
            self.history.append(self.current_move_index)
            self.future = []  # Clear redo history when making new moves
            
            self.current_move_index += 1
            self.current_board, _ = self.get_current_state()
            return True
        return False
    
    def step_backward(self):
        """Move backward one step in the replay (undo)."""
        if self.current_move_index > 0:
            # Save current state to future for redo
            self.future.append(self.current_move_index)
            
            self.current_move_index -= 1
            self.current_board, _ = self.get_current_state()
            return True
        return False
    
    def jump_to_start(self):
        """Jump to the beginning of the replay."""
        if self.current_move_index > 0:
            self.history.append(self.current_move_index)
            self.future = []
            self.current_move_index = 0
            self.current_board, _ = self.get_current_state()
    
    def jump_to_end(self):
        """Jump to the end of the replay."""
        if self.current_move_index < len(self.moves) - 1:
            self.history.append(self.current_move_index)
            self.future = []
            self.current_move_index = len(self.moves) - 1
            self.current_board, _ = self.get_current_state()
    
    def undo(self):
        """Undo the last navigation action."""
        if self.history:
            prev_index = self.history.pop()
            self.future.append(self.current_move_index)
            self.current_move_index = prev_index
            self.current_board, _ = self.get_current_state()
            return True
        return False
    
    def redo(self):
        """Redo a previously undone navigation action."""
        if self.future:
            next_index = self.future.pop()
            self.history.append(self.current_move_index)
            self.current_move_index = next_index
            self.current_board, _ = self.get_current_state()
            return True
        return False
    
    def draw(self):
        """Draw the current board state and replay controls."""
        move_data = None
        board_to_draw = self.current_board
        
        if self.current_move_index < len(self.moves):
            move_data = self.moves[self.current_move_index]
            # Use board_after if available, otherwise board
            if 'board_after' in move_data:
                board_to_draw = np.array(move_data['board_after'])
            elif 'board' in move_data:
                board_to_draw = np.array(move_data['board'])
        
        # Draw board
        valid_moves = []
        if move_data and 'valid_moves' in move_data:
            valid_moves = move_data['valid_moves']
        
        draw_board(board_to_draw, valid_moves, display=True)
        
        # Draw status information
        whites, blacks = count_discs(board_to_draw)
        status_text = f"Move: {self.current_move_index}/{len(self.moves) - 1} | W:{whites} B:{blacks}"
        text_surface = self.font.render(status_text, True, BLACK, GRAY)
        self.screen.blit(text_surface, (10, WINDOW_SIZE + 10))
        
        # Draw move information
        if move_data:
            if move_data.get('game_over'):
                winner_text = f"Game Over! Winner: {move_data.get('winner', 'Unknown')}"
            elif move_data.get('move'):
                player_name = move_data.get('player', 'Unknown').capitalize()
                move_pos = move_data.get('move', ())
                move_text = f"{player_name} played: ({move_pos[0]}, {move_pos[1]})"
            else:
                move_text = "Initial position"
            
            if move_data.get('game_over'):
                winner_surface = self.font.render(winner_text, True, BLACK, GRAY)
                self.screen.blit(winner_surface, (10, WINDOW_SIZE + 50))
            elif move_data.get('move'):
                move_surface = self.small_font.render(move_text, True, BLACK)
                self.screen.blit(move_surface, (10, WINDOW_SIZE + 50))
        
        # Draw controls help
        controls = [
            "Controls:",
            "←/→ : Step backward/forward",
            "Home/End : Jump to start/end",
            "U/R : Undo/Redo navigation",
            "Q/ESC : Quit"
        ]
        y_offset = WINDOW_SIZE + 80
        for i, control in enumerate(controls):
            if i == 0:
                control_surface = self.small_font.render(control, True, BLACK, GRAY)
            else:
                control_surface = self.small_font.render(control, True, BLACK)
            self.screen.blit(control_surface, (WINDOW_SIZE - 250, y_offset + i * 20))
        
        pygame.display.flip()
    
    def run(self):
        """Run the replay viewer main loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT or event.key == pygame.K_SPACE:
                        self.step_forward()
                    elif event.key == pygame.K_LEFT:
                        self.step_backward()
                    elif event.key == pygame.K_HOME:
                        self.jump_to_start()
                    elif event.key == pygame.K_END:
                        self.jump_to_end()
                    elif event.key == pygame.K_u:
                        self.undo()
                    elif event.key == pygame.K_r:
                        self.redo()
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
            
            self.draw()
            clock.tick(60)
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Othello Replay Viewer")
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Replay file (JSON format)"
    )
    args = parser.parse_args()
    
    try:
        viewer = ReplayViewer(args.file)
        viewer.run()
    except FileNotFoundError:
        print(f"Error: Replay file '{args.file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in replay file '{args.file}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

