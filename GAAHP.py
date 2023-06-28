import numpy as np
import random


POP_SIZE = 100
CROSSOVER_RATE = 0.6
FERTILITY_RATE = 1.2
MUTATION_RATE = 0.01
ELITE = 0.02
N_GENERATIONS = 100

DNA_SIZE = 0
for i in range(1,MATRIX_SIZE):
	DNA_SIZE = DNA_SIZE+i



def compute_consistency_ratio(matrix):
    shape = matrix.shape
    size = shape[0]
    results = np.zeros(size)
    ri_dict = {3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45,
                10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}
    if MATRIX_SIZE < 3:
        return results
    else:
        for n in range(size):
            # Find the Perron-Frobenius eigenvalue of the matrix
            _matrix = matrix[n,:,:]

            random_index = ri_dict[MATRIX_SIZE]
            lambda_max = np.max(np.linalg.eigvals(_matrix))
            consistency_index = (lambda_max - MATRIX_SIZE) / (MATRIX_SIZE - 1)
            # The absolute value avoids confusion in those rare cases where a small negative float is rounded to -0.0
            #consistency_ratio = np.abs(np.real(consistency_index / random_index).round(self.precision))
            results[n] = np.abs(np.real(consistency_index / random_index).round(3))
        return results


def get_fitness(pop):
    #x, y = translateDNA(pop)
    allMatrix = translateDNA(pop)
    #pred = F(x, y)
    pred = compute_consistency_ratio(allMatrix)
    return pred
    # return pred - np.min(pred)+1e-3
    # return np.max(pred) - pred + 1e-3


def translateDNA(pop):
    #x_pop = pop[:, 0:DNA_SIZE]
    #y_pop = pop[:, DNA_SIZE:] 
    #x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    #y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    shape = pop.shape
    size = shape[0]
    #print(size)
    allMatrix = np.zeros((size, MATRIX_SIZE, MATRIX_SIZE))
    for n in range(size):
        matrix = np.zeros((MATRIX_SIZE,MATRIX_SIZE))
        index = 0
        for i in range(MATRIX_SIZE):
            for j in range(i, MATRIX_SIZE):
                if i == j:
                    matrix[i, j] = 1
                else:
                    if pop[n, index] > 0:
                        value = pop[n, index]
                    elif pop[n, index] < 0:
                        value = 1 / abs(pop[n, index])
                    matrix[i, j] = value
                    matrix[j, i] = 1 / value
                    index += 1
        allMatrix[n,:,:] = matrix

    return allMatrix


def crossover_and_mutation(pop, fitness, CROSSOVER_RATE=0.8):
    new_pop = []
    numElite = int(np.round(POP_SIZE*ELITE))
    #for father in pop:
    
    min_indices = np.argpartition(fitness, numElite)[:numElite]
    for n in range(numElite):
        new_pop.append(pop[min_indices[n],:])

    for n in range(numElite, int(POP_SIZE*FERTILITY_RATE)):
        mutate_point = np.random.randint(0, POP_SIZE)
        father = pop[mutate_point,:]
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = pop[np.random.randint(POP_SIZE)]
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)
            child[cross_points:] = mother[cross_points:]
        mutation(child)
        new_pop.append(child)

    return new_pop

def select(pop, fitness):
    num = round(POP_SIZE * (FERTILITY_RATE - 1.0))
    max_indices = np.argpartition(fitness, -num)[-num:]
    #print(max_indices)
    #pop = np.delete(pop, max_indices, axis = 0)
    return np.delete(pop, max_indices, axis = 0)



def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:
        mutate_point = np.random.randint(0, DNA_SIZE)
        if child[mutate_point] == 1:
            child[mutate_point] = 2
        elif child[mutate_point] == -1:
            child[mutate_point] = -2
        elif child[mutate_point] == 9:
            child[mutate_point] = 8
        elif child[mutate_point] == -9:
            child[mutate_point] = -8
        else:
            probability = random.random()
            if probability < 0.5:
                child[mutate_point] += 1
            else:
                child[mutate_point] -= 1



def print_info(pop):
    fitness = get_fitness(pop)
    min_fitness_index = np.argmin(fitness)
    print("min_fitness:", fitness[min_fitness_index])
    #x, y = translateDNA(pop)
    print("best chromosome：", pop[min_fitness_index])


if __name__ == "__main__":

    pop = np.loadtxt("matrixC.dat")
    print_info(pop)

    for _ in range(N_GENERATIONS):
        fitness = get_fitness(pop)
        pop = np.array(crossover_and_mutation(pop, fitness, CROSSOVER_RATE))
        fitness = get_fitness(pop)
        pop = select(pop, fitness)

    print_info(pop)