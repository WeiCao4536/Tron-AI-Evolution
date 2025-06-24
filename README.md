Tron Light Cycle AI - Project Documentation
Overview
A genetic algorithm-powered neural network system that learns to play the classic Tron Light Cycle game through evolutionary training. The AI agents compete in a grid-based environment, developing strategies through natural selection.

Key Components
1. Genetic Neural Network
Structure: Input layer (40 nodes), hidden layers (16, 8 nodes), output layer (4 nodes)

Activation: ReLU for hidden layers, Softmax for output

Learning: Evolves through genetic operations rather than backpropagation

2. Evolutionary System
Population Size: Configurable (default 100)

Selection: Tournament selection from top performers

Genetic Operations:

Crossover (50% probability)

Mutation (16% rate with variable strength)

Elite preservation (20%)

3. Game Environment
Grid: 30x20 cell arena

Players: 4 AI agents compete simultaneously

Movement: UP/DOWN/LEFT/RIGHT actions

Training Process
Evaluation: Each AI plays multiple games against random opponents

Fitness Scoring: Based on:

Survival time (primary)

Movement diversity (secondary)

Action balance (secondary)

Reproduction: Best performers breed to create next generation

Termination: After specified generations (default 20,000)

Usage
Running the Simulation
python
final_population, best_network = genetic_algorithm(
    population_size=100,
    generations=20000,
    games_per_ai=2,
    elite_ratio=0.2,
    mutation_rate=0.16
)
Visualizing Results
python
env = VisualTronEnvironment()
env.run_ai_match(best_network)  # Watch the best AI play
File Structure
game_env.py - Core game logic and environment

genetic_nn.py - Neural network implementation

game_screen.py - Pygame rendering system

saved_population.pkl - Trained model storage

Requirements
Python 3.7+

NumPy

Pygame

Pickle
