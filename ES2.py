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

enemylist = [3,6,8]
experiment_name = f'ES2_{enemylist}'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# The NN has one hidden layer with 10 neurons
n_hidden_neurons = 10
# Initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=enemylist,
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)
# Default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

class EA2:
    def __init__(self, pop_size, n_generations, n_runs, mutation_rate, learning_rate, boundary, alpha):
        self.n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
        self.upper_bound = 1
        self.lower_bound = -1
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.n_runs = n_runs
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.boundary = boundary
        self.alpha = alpha

    def initialize(self):
        '''
        Initializes a population of pop_size individuals with the sigma value to the last index of each individual
        Returns the population with  the sigma values
        '''
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.n_weights+1))
        
        # Set bias of jump output node to high value
        # 210 = left, 211 = right, 212 = jump, 213 = shoot, 214 = release
        for i in range(pop.shape[0]):
            pop[i, 213] = 1000 # shoot
            pop[i, 212] = 1000 # jump

        return pop

    def simulation(self, x):
        """
        Run a simulation for one individual in the population.
        Parameters:
            x (array-like): The controller or policy for the individual.
        Returns:
            float: The fitness score and gain of the individual.
        """
        # Run the simulation and get fitness, player energy, enemy energy, and duration
        # For simulation, exclude sigma values
        individual = x[:-1] 
        fit, p_energy, e_energy, duration = env.play(pcont=individual)
        gain = p_energy - e_energy
        return fit, gain

    def evaluate(self, pop):
        """
        Determine the fitnesses of individuals in the population.
        Parameters:
            pop (list): The population of individuals.
        Returns:
            numpy.ndarray: An array containing the fitness score and gain for each individual.
        """
        pop_fit_gain = np.array([self.simulation(y) for y in pop])
        return pop_fit_gain

    def random_uniform_parent_selection(self, pop):
        """
        Chooses randomly parents
        Parameters:
            pop (numpy.ndarray): The population of individuals.
        Returns:
            List with two parents that are not the same individuals so we prevent reproduction with themselves
        """
        n_individuals = pop.shape[0]
        parent1_index = 0
        parent2_index = 0
        while parent1_index == parent2_index:
            parent1_index = np.random.randint(0, n_individuals - 1)
            parent1 = pop[parent1_index]
            parent2_index = np.random.randint(0, n_individuals - 1)
            parent2 = pop[parent2_index]
        parents = [parent1, parent2]

        return parents


    def whole_arithmic_crossover(self, pop):
        """
        Perform whole arithmetic crossover on a population.
        Parameters:
            pop (numpy.ndarray): The population of individuals.
        Returns:
            numpy.ndarray: The resulting offspring population.
        """
        offspring = [] 
        # "λ is typically much higher than μ (recently values around 1/4 seem to gain popularity)" --> so from a population of 100 individuals, we get 400 children, so have to do the crossover 400 times
        for _ in range(400): 
            parents = self.random_uniform_parent_selection(pop)
            parent1 = parents[0]
            parent2 = parents[1]
            child1 = [] 
            for gene1, gene2 in zip(parent1, parent2):
                offspring1 = self.alpha * gene1 + (1 - self.alpha) * gene2 # 2 parents only create 1 child
                child1.append(offspring1)
            offspring.extend([child1])
        return np.array(offspring)

    def limits(self, x):
        """
        Ensure x is within specified bounds.
        Parameters:
            x (float): The input value.
        Returns:
            float: The bounded value.
        """
        if x > self.upper_bound:
            return self.upper_bound
        elif x < self.lower_bound:
            return self.lower_bound
        else:
            return x
        
    def update_sigma(self, offspring):
        """
        Update the value of sigma.
        Args:
            offspring (numpy.ndarray): an individual offspring
        Returns:
            float: The updated value of sigma.
        """
        exponent = np.exp(self.learning_rate * (np.random.normal(0, 1)))
        offspring[-1] *= exponent
        if offspring[-1] < self.boundary:
            offspring[-1] = self.boundary
        return offspring

    def self_adapt_mutate(self, offspring):
        """
        Apply self-adaptive mutation with one step size to the offspring.
        Args:
            offspring (numpy.ndarray): The offspring population.
        Returns:
            numpy.ndarray: The mutated offspring population.
        """
        # Updating sigma values of the offspring
        offspring = self.update_sigma(offspring)
        # Exclude sigma in the range
        for i in range(len(offspring)-1):
            if np.random.uniform(0, 1) <= self.mutation_rate:
                offspring[i] += offspring[-1] * np.random.normal(0, 1)
        offspring = np.array([self.limits(y) for y in offspring])
        return offspring

    def survival_selection(self, offspring):
        """
        Perform (μ,λ) survival selection to create the next generation.
        Args:
            offspring (numpy.ndarray): The offspring population.
        Returns:
            numpy.ndarray: The population of the new generation.
        """
        # Get the fitness scores of the offspring
        new_pop_fit_gain = self.evaluate(offspring)
        new_pop_fit = new_pop_fit_gain[:,0]
        
        # Sort the total offsprinf by fitness (in descending order) and get the indices of the best 100 individuals
        #this way ther 100 best offspring of the 400 are kept 
        sorted_fit_indices = np.argsort(new_pop_fit)[::-1]
        best_indices = sorted_fit_indices[:100]
        
        # Select the best individuals and their fitness scores
        best_pop = [offspring[i] for i in best_indices]
        return np.array(best_pop)

# when importing this file for the boxplots we do not run anything below this
EA = EA2(100,     30,            10,     0.01,         0.1,           0.001,     0.9)
#(self, pop_size, n_generations, n_runs, mutation_rate, learning_rate, boundary, alpha)
attributes = vars(EA)
with open(f'{experiment_name}/params.txt', "w") as file:
    for key, value in attributes.items():
        file.write(f"{key}: {value}\n")

if __name__ == "__main__": 
    indices_run     = []
    indices_gen     = []
    
    best_gain       = []
    best_fit        = []
    mean_fitness    = []
    std_fitness     = []
    best_solutions  = []

    # EVOLUTIONARY LOOP
    for r in range(EA.n_runs):
        i = 0
        pop = EA.initialize()
        pop_fit_gain = EA.evaluate(pop)
        pop_fit = pop_fit_gain[:,0]
        pop_gain = pop_fit_gain[:,1]
        
        best = np.argmax(pop_fit)
        best_solution = pop[best].tolist()
        mean = np.mean(pop_fit)
        std = np.std(pop_fit)

        # Saves result
        print( '\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(pop_fit[best],6))+'  '+str(round(mean,6))+' '+str(round(std,6)))    
        experiment_data  = open(experiment_name+'/results.txt','a')
        experiment_data.write('\n RUN '+str(r)+' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(pop_fit[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))
        experiment_data.close()

        indices_run.append(r)
        indices_gen.append(i)
        
        best_gain.append(pop_gain[best])
        best_fit.append(pop_fit[best])
        mean_fitness.append(mean)
        std_fitness.append(std)
        best_solutions.append(best_solution)

    # Loop through generations
        for i in range(1,EA.n_generations):
            # Create offspring applying crossover and mutation
            offspring = EA.whole_arithmic_crossover(pop)
            offspring = [EA.self_adapt_mutate(child) for child in offspring]
            
            # Survival selection (10 elite parents + 90 random children)
            pop = EA.survival_selection(offspring)

            pop_fit_gain = EA.evaluate(pop)
            pop_fit = pop_fit_gain[:,0]
            pop_gain = pop_fit_gain[:,1]

            best = np.argmax(pop_fit)
            best_solution = pop[best].tolist()
            std  =  np.std(pop_fit)
            mean = np.mean(pop_fit)

            # Saves result
            print('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(pop_fit[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))        
            experiment_data  = open(experiment_name+'/results.txt','a')
            experiment_data.write('\n RUN '+str(r)+ ' GENERATION '+str(i)+'  '+str(round(pop_gain[best],6))+'  '+str(round(pop_fit[best],6))+'  '+str(round(mean,6))+'  '+str(round(std,6)))
            experiment_data.close()

            indices_run.append(r)
            indices_gen.append(i)
            
            best_gain.append(pop_gain[best])
            best_fit.append(pop_fit[best])
            mean_fitness.append(mean)
            std_fitness.append(std)
            best_solutions.append(best_solution)

    d = {"Run": indices_run, "Gen": indices_gen, "gain": best_gain, "Best fit": best_fit, "Mean": mean_fitness, "STD": std_fitness,"BEST SOL":best_solutions}
    df = pd.DataFrame(data=d)
    print(df)
    # makes csv file
    df.to_csv(f'{experiment_name}\{experiment_name}.csv', index = False)
    