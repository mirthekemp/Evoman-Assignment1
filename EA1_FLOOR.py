################################
# EvoMan FrameWork - EA1       #
# Author: Group 65             #
#                              #
################################

# Import framwork and other libs
import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import csv


# Choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

enemy = 7  # MAKE SURE TO ALSO CHANGE LINE 36
# Create a folder for the experiment in which all the data are stored
experiment_name = f'EA1_enemy{enemy}'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# The NN has one hidden layer with 10 neurons
n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# Default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

# OPTIMIZATION FOR CONTROLLER SOLUTION: GENETIC ALGORITHM

# Genetic algorithm parameters
n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 # number of weights for multilayer with 10 hidden neurons
upper_bound = 1 # upper bound of start weights
lower_bound = -1 # lower bound of start weights
pop_size = 100
n_generations = 30
n_runs = 10
mutation_rate = 0.01
k = 10

def initialize(population_size, lower_bound, upper_bound, n_weights):
    return np.random.uniform(lower_bound, upper_bound, (population_size, n_weights))

def simulation(env,x):
    """
    Run a simulation for one individual in the population.

    Parameters:
        env (Environment): The environment object representing the game.
        pop (array-like): The controller or policy for the individual.

    Returns:
        float: The fitness score and gain of the individual.
    """
    # Run the simulation and get fitness, player energy, enemy energy, and duration
    fit,p_energy,e_energy,duration = env.play(pcont=x)
    gain = p_energy - e_energy
    return fit, gain

def evaluate(pop):
    """
    Determine the fitnesses of individuals in the population.

    Parameters:
        pop (list): The population of individuals.

    Returns:
        numpy.ndarray: An array containing the fitness score for each individual.
    """
    pop_fit_gain = np.array([simulation(env, y) for y in pop])
    return pop_fit_gain

def tournament(pop, pop_fit, k):
    """
    Perform a tournament selection on a population.

    Parameters:
        pop (numpy.ndarray): The population of individuals.
        pop_fit (numpy.ndarray): The fitness scores of the individuals.
        k (int): The number of individuals competing in each tournament.

    Returns:
        numpy.ndarray: The winning individual from the tournament.
    """
    n_individuals = pop.shape[0]
    current_winner = np.random.randint(0, n_individuals-1)
    current_max_fit = pop_fit[current_winner]

    for candidates in range(k-1): #We already have one candidate, so we are left with k-1 to choose
        contender_number = np.random.randint(0, n_individuals-1)
        if pop_fit[contender_number] > current_max_fit:
            current_winner = contender_number
            current_max_fit = pop_fit[contender_number]
    winner = pop[current_winner]
    return winner

def whole_arithmic_crossover(pop, pop_fit, k, alpha=0.5):
    """
    Perform whole arithmetic crossover on a population.

    Parameters:
        pop (numpy.ndarray): The population of individuals.
        pop_fit (numpy.ndarray): The fitness scores of the individuals.
        k (int): The number of individuals competing in each tournament.
        alpha (float, optional): The blending factor. Default is 0.5.

    Returns:
        numpy.ndarray: The resulting offspring population.
    """
    offspring = []
    for p in range(0, pop.shape[0], 2):
        parent1 = tournament(pop, pop_fit, k)
        parent2 = tournament(pop, pop_fit, k)
        child1 = []
        child2 = []
        for gene1, gene2 in zip(parent1, parent2):
            offspring1 = alpha * gene1 + (1 - alpha) * gene2
            offspring2 = alpha * gene2 + (1 - alpha) * gene1
            child1.append(offspring1)
            child2.append(offspring2)
        offspring.extend([child1, child2])
    return np.array(offspring)

def uniform_mutation(offspring, mutation_rate):
    """
    Apply unifom mutation to the offspring.

    Args:
        offspring (numpy.ndarray): The offspring population.
        mutation_rate (float): The mutation rate.
    Returns:
        numpy.ndarray: The mutated offspring population.
    """
    for i in range(len(offspring)):
        if np.random.uniform(0,1) <= mutation_rate:
            offspring[i] += np.random.uniform(-1, 1)
    return offspring

def elitism(pop, pop_fit, x):
    """
    Select the best x individuals from the population based on fitness.

    Parameters:
        pop (list): The list of individuals.
        pop_fit (list): The fitness scores of the individuals.
        pop_gain (list): The gain of the individuals.
        x (int): The number of best individuals to select.

    Returns:
        tuple: A tuple containing 2 lists - best individuals, their fitness scores
    """
    # Sort the population by fitness (in descending order) and get the indices of the best individuals
    sorted_fit_indices = np.argsort(pop_fit)[::-1]
    best_indices = sorted_fit_indices[:x]
    
    # Select the best individuals and their fitness scores and gains
    best_pop = [pop[i] for i in best_indices]
    best_pop_fit = [pop_fit[i] for i in best_indices]

    return best_pop, best_pop_fit

def elitism_survival_selection(parents, parents_fit, offspring, x, y):
    """
    Perform survival selection to create the next generation.

    This function combines the fittest parents and randomly selected children to form the new population.

    Parameters:
        parents (numpy.ndarray): The parent individuals.
        parents_fit (numpy.ndarray): The fitness scores of the parents.
        offspring (numpy.ndarray): The child individuals.
        x (int): The number of fittest parents to keep.
        y (int): The number of random children to keep.

    Returns:
        numpy.ndarray: The new population.
    """
    # Check if x and y are correct values
    if (x + y) != 100 or x < 0 or y < 0 or x > 100 or y > 100:
        raise ValueError("The values of x and y are incorrect.")
    
    # Select the x fittest parents
    best_parents, best_parents_fit = elitism(parents, parents_fit, x)

    # Select y children from offspring
    random_indices = random.sample(range(len(offspring)), y)
    selected_offspring = [offspring[i] for i in random_indices]

    # Combine the best parents and random children to form the new population
    pop = np.vstack((best_parents, selected_offspring))

    return pop

# when importing this file for the boxplots we do not run anything below this
if __name__ == "__main__": 
    indices_run     = []
    indices_gen     = []
    
    best_gain       = []
    best_fit        = []
    mean_fitness    = []
    std_fitness     = []
    best_solutions  = []
    game_lostwon    = []

    result_matrix_max=np.zeros((n_runs,n_generations))
    result_matrix_mean=np.zeros((n_runs,n_generations))
    
    # EVOLUTIONARY LOOP
    for r in range(n_runs):
        i = 0
        pop = initialize(pop_size, lower_bound, upper_bound, n_weights)
        pop_fit_gain = evaluate(pop)
        pop_fit = pop_fit_gain[:,0]
        pop_gain = pop_fit_gain[:,1]
        
        best = np.argmax(pop_fit)
        best_solution = pop[best].tolist()
        mean = np.mean(pop_fit)
        std  = np.std(pop_fit)
        
        # Saves result
        print('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(pop_fit[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))    
        #experiment_data  = open(experiment_name+'/results.txt','a')
        #experiment_data.write('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(pop_fit[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))
        #experiment_data.close()

        result_matrix_max[r,i]=np.max(pop_fit)
        result_matrix_mean[r,i]=np.mean(pop_fit)
        
        indices_run.append(r)
        indices_gen.append(i)
        
        best_gain.append(pop_gain[best])
        best_fit.append(pop_fit[best])
        mean_fitness.append(mean)
        std_fitness.append(std)
        best_solutions.append(best_solution)

    # Loop through generations
        for i in range(1,n_generations):
            # Create offspring applying crossover and mutation
            offspring = whole_arithmic_crossover(pop, pop_fit, k, alpha=0.5)
            offspring = [uniform_mutation(gene, mutation_rate) for gene in offspring]
            
            # Survival selection (10 elite parents + 90 random children)
            pop = elitism_survival_selection(pop, pop_fit, offspring, 1, 99)
            pop_fit_gain = evaluate(pop)
            pop_fit = pop_fit_gain[:,0]
            pop_gain = pop_fit_gain[:,1]

            best = np.argmax(pop_fit)
            best_solution = pop[best].tolist()
            mean = np.mean(pop_fit)
            std  =  np.std(pop_fit)

            # Saves result
            print('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(pop_fit[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))    
            #experiment_data  = open(experiment_name+'/results.txt','a')
            #experiment_data.write('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(pop_fit[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))
            #experiment_data.close()

            result_matrix_max[r,i]=np.max(pop_fit)
            result_matrix_mean[r,i]=np.mean(pop_fit)

            indices_run.append(r)
            indices_gen.append(i)
            
            best_gain.append(pop_gain[best])
            best_fit.append(pop_fit[best])
            mean_fitness.append(mean)
            std_fitness.append(std)
            best_solutions.append(best_solution)

            # #check if best solution of generation wins the game:
            # if i == n_generations-1:
            #     fit,p_energy,e_energy,duration = env.play(pcont=pop[best])
            #     if e_energy ==0:
            #         game_lostwon.append('won')
            #     else:
            #         game_lostwon.append('lost')
    d = {"Run": indices_run, "Gen": indices_gen, "gain": best_gain, "Best fit": best_fit, "Mean": mean_fitness, "STD": std_fitness, "BEST SOL": best_solutions}
    df = pd.DataFrame(data=d)
    # num_rows = len(df)
    # # Fill the DataFrame with values from the list every 30th row
    # for i in range(0, num_rows):
    #     if i % 30 == 29 and i // 30 < len(game_lostwon):
    #         df.loc[i, 'lost/won'] = game_lostwon[i // 30]
    print(df)
    #makes csv file
    #df.to_csv(f'{experiment_name}\{experiment_name}.csv', index=False)
    
