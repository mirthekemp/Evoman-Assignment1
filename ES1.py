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
experiment_name = f'ES1_{enemylist}_floor'
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

class EA1:
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
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size-2, self.n_weights+1))
        seeded_individual1 =  np.array([0.11260463189438405, -0.17554375127073918, 0.18364108735962192, 0.19976281087751638, -0.13176632677336375, 0.11124750412189835, -0.18138965344453528, -0.026317928013631378, 0.049317010933686, -0.2992875928654357, 0.20825958538532233, 0.08394919726179426, -0.15733115945124865, 0.2588717445419593, -0.029764828446903074, 0.08030231292293713, 0.19673743909307284, 0.44572558716681715, 0.21332053908643384, 0.12016655284794055, 0.09363925720474109, 0.13041074896667665, -0.3837921636717464, -0.49860056330257385, 0.2518040888919044, 0.011042850068441624, -0.031640386402938606, -0.09534925399759783, 0.06297732569410044, -0.03530531673566148, 0.018748615560833085, 0.036646066454670806, 0.05923383537320981, 0.20787320714462548, -0.183543565945194, 0.11452955261035956, -0.09311880696304198, -0.14502355429131397, 0.04805092281445533, 0.2695164177700682, 0.20866797414076177, 0.18526681437432235, -0.06084726317803775, -0.2575917574723955, -0.04846027472151276, 0.07613868775464785, -0.09733741654260045, -0.057524010210289675, 0.2705504669746731, 0.205491478140908, -0.4185800551614971, 0.20625209750430448, 0.41019181653520975, 0.09136500040103947, -0.020418443024714877, -0.09423575684910732, 0.05122896912657679, -0.28356943408745494, -0.344174349978715, -0.00724213886090733, 0.4508651438351457, -0.21701128436492534, -0.4304691075266142, -0.19474328495148036, 0.17182182801341345, -0.07871710062589654, -0.01939343316565355, 0.07721227732504003, -0.20809070768118942, -0.2211551070448629, 0.10989886220434728, -0.06994588834757981, 0.07420842842092742, -0.21861261508058794, 0.22362724659349306, -0.1650994730539015, -0.06754333974800968, 0.18480493244291857, -0.14190475436246908, -0.09206959195913575, -0.12979417565803356, 0.12527399555526314, -0.028856825465106813, -0.10582526801419886, 0.17297304410438613, 0.13249528403166436, -0.28217726300916857, 0.06228580961500664, -0.3898262173585587, -0.24937381595052444, -0.050774832811388594, -0.13737557440639542, -0.14938573054828536, -0.1465588813883976, -0.27874673416468443, 0.12491487264028103, 0.19344382838074864, -0.06659272608818693, 0.21574731104039696, 0.09926779623285838, 0.36933200844311487, -0.0681998278747051, 0.08737538419862223, 0.22162210860559733, 0.11325682554315952, 0.18111101216478792, 0.15568532918496225, 0.20152234812442738, 0.07433643844411345, -0.14432509787174047, 0.03079954574996826, -0.06287081257464855, -0.2430287918492276, 0.13862371616954394, -0.3169321086005202, 0.12167399736584908, -0.05719097026856276, 0.042378119556245146, 0.32626530098633877, -0.29807223346643374, 0.2734192964427723, 0.3200777118648529, -0.0912514526489532, 0.1043728453908469, 0.3592685414598871, 0.15658604623260702, -0.0019827494194118167, -0.20274025816256197, -0.0994119290850138, 0.06474942582212417, -0.41880576723917906, 0.20837846122216622, 0.1551194018214449, 0.01539019706030536, -0.2270139709758652, 0.21177260693065347, 0.1666555767912126, 0.1323530056272735, -0.19352689204171905, -0.1731057358243164, -0.01977463564471847, -0.11312677247131293, -0.2879975222200697, -0.16848171247306398, -0.007069642782140237, 0.20743590405351442, 0.29842016880144473, -0.12283134024336852, 0.23327326107190868, 0.22397480308035944, -0.30264816994472565, -0.19208356675313787, 0.26233896230468234, -0.04564154934582004, 0.0677827546680973, -0.019045957627733163, -0.3286922242110429, 0.1877590435836578, 0.04173048275157512, 0.12965806363314764, -0.11331953299685048, 0.10664322959634942, -0.06087048693748262, 0.2492432697214593, -0.08840698418533355, 0.26355275360506575, -0.1634968198502207, 0.1517625630112706, -0.2044043753661821, 0.18174829326341416, -0.18372757562856412, -0.22100777616808093, -0.24734872535846505, 0.10783804215638529, -0.4808353405792505, -0.09155236985595278, -0.18239517963496993, -0.17455786368700493, -0.0514541379962896, -0.046619755305584695, -0.09532928740148873, 0.1797516644008975, -0.16472364205714032, -0.13910164610454134, 0.11725934433146218, -0.03795793717360467, 0.23178095751540215, 0.061739080520384924, -0.024987325775022565, 0.27097479016312837, -0.041958005516863586, -0.28042860268598074, 0.404801881139095, -0.1856325684416035, 0.09342721023490994, 0.2333318201840146, 0.06621809954671365, 0.0376451012975466, 0.11929968624979195, -0.1216327282080736, 0.12989343561250827, 0.20376665947655156, -0.149347903997411, -0.3578842324197382, 0.32181383611221825, -0.21716257356223895, 0.08720128651200602, 0.1577375521156366, 0.09204973431641834, 0.0576765352460281, -0.19713382071980493, -0.2929483713651626, 1.0, 1.0, 0.16375850829664163, -0.36695571432574214, 0.20619945409113488, 0.2962438926023545, -0.042252805400003764, -0.05198383541009558, -0.3732047004397565, -0.10122775186923963, 0.23014043341049428, -0.32010951528357867, 0.12663435291011207, 0.3154809042019002, -0.40175812064634797, 0.05220732506561087, -0.1663484993479793, -0.11277351019978234, -0.1496898137304783, -0.11357479087606928, 0.14431815639054862, -0.2745167361288908, 0.3040567966029211, -0.07642748390436596, 0.07937176167528387, 0.23459269026729096, 0.19990401471506022, -0.022749676208189858, -0.3716756833897712, 0.24396111567774084, 0.12282900634439405, 0.11063564016704142, 0.2067041331964639, -0.016645624212484854, -0.06193926199079816, -0.19790539309377403, 0.19993607297862082, -0.019988940352994797, 0.3241383533622728, 0.47416739344358355, -0.09002443649965254, 0.284094552043907, 0.19197657626421447, 0.22042694669065768, -0.0208431460188338, -0.23428734063873008, 0.1302946033999343, -0.20284349687959602, 0.1907485316687838, -0.07449324279253766, -0.00902045113738526, 0.18253580384687057, 0.20962983627209025, 0.04802323554506822]).reshape(1, 266)
        seeded_individual2 = np.array([-0.21742866521138554, -0.09574827584564038, -0.3606133599735535, 0.1094717433995011, -0.22342671707750017, 0.28426728040873456, 0.035582407965907586, 0.1807585229252523, -0.13717152914646102, -0.088337149025599, -0.4958924157516968, 0.20715714150475234, -0.3469695601831837, -0.18805760216126755, 0.12064652227019815, -0.18438735522981353, 0.27695993188620943, 0.006980871102143268, 0.005088698918636199, -0.4807390559056856, 0.14493791601247363, -0.2512036145714718, 0.08594995578830045, 0.1775244719940111, 0.08377653736992988, -0.05115065932208561, -0.16645571034495427, 0.35530629817364057, 0.25934610819398, 0.07522850064041794, -0.11022608135400844, -0.010931509859689529, -0.12848454762851536, 0.13726815522391106, 0.09491783893110725, -0.07761013565783142, 0.1328560882713289, 0.016109382446090212, -0.11013345295602134, 0.25834075846143323, 0.1703370703134305, -0.2746668636078082, -0.2720048924294235, 0.08147063550192553, 0.24181750102470595, -0.01199659431598251, -0.1853307885276515, -0.05189664194453836, -0.20611500720865605, 0.12145029449901332, -0.23463014326285, 0.1692644929522697, -0.04455309487515639, 0.00209851149422416, 0.30382845651772217, -0.03748762734642953, -0.07972957568050847, 0.14422087212310747, -0.1448177947722591, -0.29476211589562296, 0.14020829047213854, -0.14078096013497124, -0.0448243112806946, 0.11565487196085698, 0.286116103381855, 0.14818746791474202, -0.04646175991680701, -0.036758674049320594, -0.08318173632858683, 0.04752425679009698, 0.1135426566227679, -0.25707797445689695, 0.12411993984063974, 0.22410540596353407, -0.14485623711941117, 0.13708783341158576, -0.0197706376886463, 0.18722578954953129, 0.07263049884369005, 0.051994280243378564, 0.23104758074864806, -0.14607494812923447, -0.020899499775790475, -0.15227588876678722, -0.0828391011159128, 0.2462508310619806, 0.16331773878594374, 0.048717191705449164, 0.37198436509921984, -0.28635915375617915, -0.08993612283219263, 0.19728123008969023, -0.3410634255641405, 0.1122921204743667, 0.05007895156318902, -0.32147865973734757, -0.18726342730295276, 0.046716925482809146, -0.37368039290405597, -0.3430350748691851, -0.22786099647649205, -0.48595602591385456, -0.022221383297626882, 0.0997056493576676, -0.3676384224523545, 0.07816098859741358, -0.1652813125616451, 0.08323458738807529, 0.20731838738681074, -0.08979337419109774, -0.27265961278603357, -0.008408974616286837, -0.3020892277521771, -0.2559059112761091, -0.18392672321345122, 0.1335891745653632, -0.11841478789210315, 0.050267606434051516, -0.09875510192140527, 0.05127049159586981, -0.2056315447934102, -0.11647352699322724, -0.08745680337959462, -0.2791960819879651, 0.11772994518610026, -0.33237748183333315, -0.2972682881676755, 0.032093338528355594, -0.001636206213581052, 0.32442613497535405, 0.2797911525776582, 0.2509196829208377, -0.07275343562517647, 0.008804305165926013, -0.0229466023208625, 0.10330905347798947, 0.0076949506873618375, 0.06407092738256895, -0.33980887496859585, -0.018101333378293283, -0.3964753731811483, -0.08010298163350432, -0.10374856419096065, 0.07254211496376926, 0.023362968517119713, 0.3551186061220913, 0.08486451562374737, 0.15891282337142568, -0.07780491343730848, -0.1977386765949899, -0.012477458052331565, 0.33003703956099695, -0.048299211423512886, -0.0069845818516403, -0.09913981133705216, 0.17036785748163688, -0.08692085567632396, 0.1489493057920399, -0.29597181456285343, -0.28656208249829074, -0.035448696710391706, 0.08593760305132334, 0.09876730513751161, -0.21991227008155995, -0.10634917124866623, -0.1647948069199066, -0.13408318185246035, -0.23000943396675322, -0.0787449762681834, 0.023282986602439864, 0.04003215077886704, -0.16841143648803578, 0.11766900826461649, 0.1294306995730462, 0.18736325474374166, -0.004532434885653859, -0.015596273657146633, 0.05101605891897158, 0.29548588325522085, -0.1337848475438915, -0.005170878437532231, -0.23009804832190267, 0.21499467326935784, -0.30451486984839976, -0.1992422912663498, -0.03188295319221089, 0.2988456383073367, 0.12707564476987898, 0.07496705573473564, -0.21335206060487805, 0.023061515832913076, -0.16376923197729548, -0.31865755925240213, 0.23402927446739236, -0.09065835835511829, -0.18204341763309095, -0.07249867074183755, 0.154541403214078, 0.24690350280082302, -0.06249936338591066, -0.14395131736294192, 0.03205975707212227, -0.19802662893673778, -0.16157340066833964, -0.16064578562427922, -0.3332038536389961, 0.05589118078886193, -0.13357652837959955, -0.04360975717163659, 0.15683339945387453, -0.3149100577139231, 0.34623568011641337, 0.999870274987827, 1.0, 0.1366925889951461, 0.022469688909972942, -0.07852327378305407, -0.1684309375535602, 0.03352901068647899, -0.09900216737550108, 0.01731507999021993, 0.0038583318078239327, -0.07965659884557785, 0.15205002617210756, -0.2567142563199928, -0.10927228346907064, 0.11783495275023789, 0.2510508457446261, -0.12929320496111218, -0.08527950769174322, -0.3271310418394844, -0.17704804836230745, 0.08770761594192447, 0.2096981666568964, 0.23717599194136837, 0.28581533616275756, 0.0034350075495836285, 0.06492985616823373, 0.015051913101534304, 0.028714873153452197, 0.068472441784046, -0.38169494289413475, 0.18084688632479856, 0.008413180908549084, -0.20965761954651074, 0.20364957556150476, -0.011618528912468788, 0.06770298920502153, -0.07514689159833449, -0.18086945212020894, -0.17667844907728256, 0.12362826914499045, -0.32970857739599, -0.09545731714435535, -0.15649271086044314, -0.17289639384094896, -0.2603294087446473, 0.11856547791324784, -0.033076739192235353, 0.34576182403357025, 0.31089162844897344, -0.41435472012553104, -0.10410177458947173, -0.1554665915847348, 0.012965494323184442, 0.10504162458537143]).reshape(1,266)
        pop = np.vstack((pop, seeded_individual1, seeded_individual2))
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

    def survival_selection(self, pop, offspring):
        """
        Perform (μ+λ) survival selection to create the next generation.
        Args:
            pop (numpy.ndarray): The original population.
            offspring (numpy.ndarray): The offspring population.
        Returns:
            numpy.ndarray: The population of the new generation.
        """
        # Combine the parents and children to form the new population
        offspring = np.array(offspring)
        new_pop = np.vstack((pop, offspring))

        # Get the fitness scores of the new population
        new_pop_fit_gain = self.evaluate(new_pop)
        new_pop_fit = new_pop_fit_gain[:,0]
        
        # Sort the total population by fitness (in descending order) and get the indices of the best 100 individuals
        sorted_fit_indices = np.argsort(new_pop_fit)[::-1]
        best_indices = sorted_fit_indices[:100]
        
        # Select the best individuals and their fitness scores
        best_pop = [new_pop[i] for i in best_indices]
        return np.array(best_pop)

# when importing this file for the boxplots we do not run anything below this
EA = EA1(100,     10,            10,     0.2,         0.1,           0.001,     0.1)
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
            pop = EA.survival_selection(pop, offspring)

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
    