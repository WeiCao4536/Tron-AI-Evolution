import pygame
import sys
import random
from game_env import TronEnvironment, Player, Point
import numpy as np
import pickle
from Ganetic_Neural_Network_for_Game_AI import GeneticNeuralNetwork

# Initialize Pygame
pygame.init()
GRID_SIZE = 20
SCREEN_WIDTH = 30 * GRID_SIZE
SCREEN_HEIGHT = 20 * GRID_SIZE
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tron Light Cycle")
clock = pygame.time.Clock()

# Color definitions
COLORS = {
    0: (200, 0, 0),    # Player 0: Red
    1: (0, 0, 200),    # Player 1: Blue
    2: (0, 200, 0),    # Player 2: Green
    3: (200, 0, 200),  # Player 3: Purple
    "bg": (0, 0, 0),   # Background: Black
    "grid": (50, 50, 50)  # Grid lines: Dark gray
}


class VisualTronEnvironment(TronEnvironment):
    def render(self):
        """Render the game screen"""
        screen.fill(COLORS["bg"])

        # Draw grid lines
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(screen, COLORS["grid"], (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(screen, COLORS["grid"], (0, y), (SCREEN_WIDTH, y))

        # Draw player trails
        for player_id, player in self.players.items():
            if not player.alive:
                continue

            color = COLORS[player_id]
            for segment in player.tail:
                pygame.draw.rect(
                    screen, color,
                    (segment.x * GRID_SIZE, segment.y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                )

            # Draw head (brighter)
            head = player.head
            pygame.draw.rect(
                screen, (min(255, color[0] + 100), min(255, color[1] + 100), min(255, color[2] + 100)),
                (head.x * GRID_SIZE, head.y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            )

        pygame.display.flip()


if __name__ == "__main__":
    # 1. Load saved population file
    with open('saved_population.pkl', 'rb') as f:
        population = pickle.load(f)

    # 2. Sort by fitness and select top 4 AIs
    top_4_ais = sorted(population, key=lambda x: -x.fitness)[:4]
    print(f"Population loaded. Using the following AIs for battle:")
    for i, ai in enumerate(top_4_ais):
        print(f"AI {i + 1}: Fitness (average survival rounds) = {ai.fitness:.1f}")

    # 3. Initialize game environment
    env = VisualTronEnvironment(num_players=4)
    clock = pygame.time.Clock()

    # 4. Victory statistics
    ai_victories = [0] * 4

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not env.is_game_over():
            # Get current player ID that needs to move
            current_player_id = env.alive_players[env.current_player_index % len(env.alive_players)]

            if env.players[current_player_id].alive:
                # Use corresponding AI decision (Player 0 = 1st AI, Player 1 = 2nd AI...)
                ai = top_4_ais[current_player_id]
                state = env.get_obs(current_player_id)
                action_probs = ai.forward(np.array([state])).flatten()
                move = ["UP", "DOWN", "LEFT", "RIGHT"][np.argmax(action_probs)]
                env.run_turn(move)

            env.render()
            clock.tick(40)  # Slightly increase game speed

        else:
            # Record match results
            winner = env.get_winner()
            if winner is not None:
                ai_victories[winner] += 1
                print(f"\nGame over! Winner: AI {winner + 1} (Player {winner})")
                print(f"Current victory statistics:")
                for i, wins in enumerate(ai_victories):
                    print(f"AI {i + 1}: {wins} wins")
            else:
                print("Game ended in a draw!")

            pygame.time.wait(2000)  # Show results for 2 seconds
            env.reset()
