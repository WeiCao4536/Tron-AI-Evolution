import numpy as np
import random
import os
import pickle
from game_env import TronEnvironment

class GeneticNeuralNetwork:
    def __init__(self, input_size=40, hidden_layers=[16, 8], output_size=4):
        """Initialize the neural network"""
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.fitness = 0  # Initial fitness set to 0
        self.game_played = 0  # New: number of survival rounds

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # Input layer to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_layers[0]) * 0.1)
        self.biases.append(np.random.randn(1, hidden_layers[0]) * 0.1)

        # Between hidden layers
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.randn(hidden_layers[i - 1], hidden_layers[i]) * 0.1)
            self.biases.append(np.random.randn(1, hidden_layers[i]) * 0.1)

        # Output layer
        self.weights.append(np.random.randn(hidden_layers[-1], output_size) * 0.1)
        self.biases.append(np.random.randn(1, output_size) * 0.1)

    def copy(self):
        """Return a completely independent copy (deep copy)"""
        new_nn = GeneticNeuralNetwork()  # Create new object
        new_nn.weights = [np.copy(w) for w in self.weights]  # Deep copy weights
        new_nn.biases = [np.copy(b) for b in self.biases]   # Deep copy biases
        new_nn.fitness = self.fitness  # Copy fitness (scalar, direct assignment)
        return new_nn

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax output layer, converts outputs to probability distribution"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def forward(self, inputs):
        layer = inputs
        for i in range(len(self.weights)):
            layer = np.dot(layer, self.weights[i]) + self.biases[i]
            if i != len(self.weights) - 1:  # If not the last layer
                layer = self.relu(layer)

        # Last layer uses softmax
        return self.softmax(layer)

    def print_weights_biases(self):
        """Print weights and biases that can be directly copied to game environment"""
        print("\n===== Directly Copyable Neural Network Parameters =====")
        print("nn.weights = [")
        for w in self.weights:
            print("    np.array([")
            for row in w:
                print("        [" + ", ".join(f"{x:.3f}" for x in row) + "],")
            print("    ]),")
        print("]")

        print("\nnn.biases = [")
        for b in self.biases:
            print("    np.array([")
            # Biases have only one row, print directly
            print("        [" + ", ".join(f"{x:.3f}" for x in b[0]) + "],")
            print("    ]),")
        print("]")
        print("===============================\n")

    def mutate(self, mutation_rate=0.12, mutation_strength=0.12):
        for i in range(len(self.weights)):
            # --- Standard additive mutation ---
            add_mask = np.random.random(self.weights[i].shape) < mutation_rate
            self.weights[i] += add_mask * np.random.normal(0, mutation_strength, self.weights[i].shape)

            # --- Multiplicative mutation (replaces large mutation functionality) ---
            # Trigger probability: 10%, strength controlled by mul_factor range
            if random.random() < mutation_rate:
                # Randomly select mutation strength mode:
                mode = random.choice(["small", "medium", "large", 'super'])
                if mode == "small":  # Small perturbation (similar to standard mutation)
                    mul_factor = np.random.uniform(0.9, 1.1, self.weights[i].shape)
                    mul_mask = np.random.random(self.weights[i].shape) < mutation_rate  # Affects 30% weights
                elif mode == "medium":  # Medium perturbation
                    mul_factor = np.random.uniform(0.8, 1.2, self.weights[i].shape)
                    mul_mask = np.random.random(self.weights[i].shape) < mutation_rate / 2  # Affects 20% weights
                elif mode == "large":  # Large perturbation (replaces original large mutation)
                    mul_factor = np.random.uniform(0.6, 1.4, self.weights[i].shape)
                    mul_mask = np.random.random(self.weights[i].shape) < mutation_rate / 4  # Affects 10% weights
                else:
                    mul_factor = np.random.uniform(0.2, 1.8, self.weights[i].shape)
                    mul_mask = np.random.random(self.weights[i].shape) < mutation_rate / 8  # Affects 10% weights

                self.weights[i] *= np.where(mul_mask, mul_factor, 1.0)

            # --- Additive mutation for biases (unchanged) ---
            bias_mask = np.random.random(self.biases[i].shape) < mutation_rate
            self.biases[i] += bias_mask * np.random.normal(0, mutation_strength, self.biases[i].shape)

def crossover(parent1, parent2):
    child = GeneticNeuralNetwork(
        input_size=parent1.input_size,
        hidden_layers=parent1.hidden_layers,
        output_size=parent1.output_size
    )

    for i in range(len(parent1.weights)):
        if random.random() < 0.5:  # 50% probability of using block crossover
            # Random block crossover
            split_row = random.randint(1, parent1.weights[i].shape[0] - 1)
            split_col = random.randint(1, parent1.weights[i].shape[1] - 1)
            child.weights[i] = np.block([
                [parent1.weights[i][:split_row, :split_col], parent2.weights[i][:split_row, split_col:]],
                [parent2.weights[i][split_row:, :split_col], parent1.weights[i][split_row:, split_col:]]
            ])
        else:  # 50% probability of using pointwise crossover
            mask = np.random.random(parent1.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])

        # Biases use pointwise crossover
        b_mask = np.random.random(parent1.biases[i].shape) > 0.5
        child.biases[i] = np.where(b_mask, parent1.biases[i], parent2.biases[i])

    return child


def save_population(population, filename='saved_population.pkl'):
    """Save current population to file"""
    with open(filename, 'wb') as f:
        pickle.dump(population, f)
    print(f"Population saved to {filename}")

def load_population(filename='saved_population.pkl'):
    """Load population from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            population = pickle.load(f)
        print(f"Loaded previous population from {filename}")
        return population
    else:
        print(f"Save file {filename} not found, using new population")
        return None


def evaluate_population(population, games_per_ai=3):
    """Each AI plays as main player for fixed number of games with random opponents,
    fitness normalized by total participation count"""
    # Initialize fitness, movement records and participation counts
    for ai in population:
        ai.fitness = 0          # Cumulative score
        ai.games_played = 0     # Record total participations (main player + opponents)
        ai.movement_history = []

    # Each AI plays as main player for games_per_ai times
    for ai in population:
        for _ in range(games_per_ai):
            # Randomly select 3 different opponents
            opponents = random.sample([p for p in population if p != ai], 3)
            players = [ai] + opponents
            random.shuffle(players)  # Randomly assign player positions

            # Run competition and record data
            rounds, movements, winner_id = run_competition(players)
            for idx, p in enumerate(players):
                p.fitness += rounds[idx]          # Cumulative survival rounds
                p.movement_history.extend(movements[idx])  # Record movement history
                p.games_played += 1               # Record participation count
                if idx == winner_id:              # Bonus points for winner
                    p.fitness += 400

    # Calculate average fitness (total score / total participations)
    for ai in population:
        if ai.games_played > 0:
            ai.fitness /= ai.games_played  # Normalize
        else:
            ai.fitness = 0  # Prevent division by zero (theoretically shouldn't happen)

        # Diversity bonus (unchanged)
        unique_directions = len(set(ai.movement_history))
        diversity_bonus = unique_directions * 40

        # Balance bonus (unchanged)
        if ai.movement_history:
            direction_counts = {}
            for move in ai.movement_history:
                direction_counts[move] = direction_counts.get(move, 0) + 1
            counts = list(direction_counts.values())
            balance_score = 1.0 / (1.0 + np.std(counts)) if len(counts) > 1 else 0
            balance_bonus = balance_score * 40
        else:
            balance_bonus = 0

        # Final fitness = average survival ability + diversity + balance
        ai.fitness += diversity_bonus + balance_bonus

    return population

def run_competition(ais):
    """Simulate a 4-player match, return survival rounds, movement records and winner ID for each AI"""
    env = TronEnvironment(len(ais))
    rounds = [0] * len(ais)
    movements = [[] for _ in range(len(ais))]  # Fixed syntax error
    turn_count = 0  # Fixed variable name

    # Create mapping from player_id to ai (correct mapping even after shuffle)
    player_to_ai = {player_id: ai for player_id, ai in enumerate(ais)}

    while not env.is_game_over():
        turn_count += 1

        # Get current alive players
        for player_id in env.alive_players:
            ai = player_to_ai[player_id]  # Find correct AI through mapping
            state = env.get_obs(player_id)

            # AI decision
            action_probs = ai.forward(np.array([state])).flatten()
            move = ["UP", "DOWN", "LEFT", "RIGHT"][np.argmax(action_probs)]

            # Record movement
            movements[player_id].append(move)

            # Execute move
            env.execute_move(player_id, move)
            rounds[player_id] += 1

    # Return winner's player_id
    winner_player_id = env.alive_players[0] if env.alive_players else -1
    return rounds, movements, winner_player_id


def genetic_algorithm(
    population_size=20,          # Population size
    generations=100,             # Number of generations
    games_per_ai=3,              # Number of evaluation games per AI
    elite_ratio=0.1,             # Elite retention ratio
    selection_ratio=0.5,         # Parent selection ratio (select from top X%)
    mutation_rate=0.1,          # Mutation rate
    mutation_strength=0.1,
    save_interval=10,           # Save interval (generations)
    crossover_prob=0.7,         # 70% individuals generated through crossover
    mutation_only_prob = 0.2,       # 20% individuals generated through mutation only
    load_previous=False          # Whether to load previous population
):

    # Initialize population
    if load_previous:
        population = load_population()

    if not load_previous or population is None:
        population = [GeneticNeuralNetwork() for _ in range(population_size)]

    for generation in range(generations):
        print(f"\n=== Generation {generation + 1} ===")

        # Automatically evaluate fitness of each individual
        population = evaluate_population(population, games_per_ai= games_per_ai)

        # Select best individuals
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        best_nn = population[0]

        print(f"\nBest fitness this generation (average survival rounds): {best_nn.fitness:.1f}")
        print("Best neural network parameters preserved for next generation")

        # Create new generation
        new_population = population[:int(population_size * elite_ratio)]

        while len(new_population) < population_size:
            rand_val = random.random()

            if rand_val < crossover_prob:  # Crossover
                top_half = population[: int(population_size * selection_ratio)]  # Select parents from top 50%
                parent1 = random.choice(top_half)
                parent2 = random.choice([p for p in top_half if p != parent1])
                child = crossover(parent1, parent2)

            elif rand_val < crossover_prob + mutation_only_prob:  # Mutation only
                top_half = population[: int(population_size * selection_ratio)]  # Select parents from top 50%
                parent1 = random.choice(top_half)
                parent2 = random.choice([p for p in top_half if p != parent1])
                child = crossover(parent1, parent2)
                child.mutate(mutation_rate, mutation_strength)

            else:  # New random individual (optional)
                child = GeneticNeuralNetwork()  # Increase diversity

            new_population.append(child)   # Add offspring

        population = new_population

        # Save every 10 generations
        if generation % save_interval == save_interval - 1:
            save_population(population)

    # Return final population and best individual
    best_nn = max(population, key=lambda x: x.fitness)
    print("\n=== Training Complete ===")
    print(f"Final best fitness (average survival rounds): {best_nn.fitness:.1f}")
    print("Best neural network parameters:")
    best_nn.print_weights_biases()
    return population, best_nn


# Usage example
if __name__ == "__main__":
    final_population, best_network = genetic_algorithm(
    population_size= 100,          # Population size
    generations= 20000,             # Number of generations
    games_per_ai= 2,              # Number of evaluation games per AI
    elite_ratio= 0.2,             # Elite retention ratio
    selection_ratio= 0.5,         # Parent selection ratio (select from top X%)
    mutation_rate= 0.16,          # Mutation rate
    mutation_strength= 0.1,
    save_interval= 5,           # Save interval (generations)
    crossover_prob= 0.5,         # 50% individuals generated through crossover
    mutation_only_prob = 0.5,       # 50% individuals generated through crossover and mutation
    load_previous= True    # Whether to load previous population
)