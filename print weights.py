from Ganetic_Neural_Network_for_Game_AI import GeneticNeuralNetwork, load_population

population = load_population()
population.sort(key=lambda x: -x.fitness)
best_nn = max(population, key=lambda x: x.fitness)

population[0].print_weights_biases()