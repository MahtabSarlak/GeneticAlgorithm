import numpy as np
import matplotlib.pyplot as plt
import random
import struct


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def mutation():
    for i in range(0, len(population)):
        probability = random.randint(0, 100000)
        if probability < 10:
            print("Mutation")
            individual = population[i]

            while True:
                index_a = random.randint(0, 31)
                index_b = random.randint(0, 31)
                index_c = random.randint(0, 31)

                parent1_a = individual[0]
                parent1_b = individual[1]
                parent1_c = individual[2]

                child1_a = parent1_a[0:index_a] + str(random.randint(0, 1)) + parent1_a[index_a + 1:len(parent1_a)]
                child1_b = parent1_b[0:index_b] + str(random.randint(0, 1)) + parent1_b[index_b + 1:len(parent1_b)]
                child1_c = parent1_c[0:index_c] + str(random.randint(0, 1)) + parent1_c[index_c + 1:len(parent1_c)]

                if ((lower_bound <= float(bin_to_float(child1_a)) <= upper_bound)) and (
                        (lower_bound <= float(bin_to_float(child1_b)) <= upper_bound)) and (
                        (lower_bound <= float(bin_to_float(child1_c)) <= upper_bound)):
                    break

            genome = [child1_a, child1_b, child1_c]
            population[i] = genome


def crossover():
    children = list()
    for i in range(0, len(population) - 1, 2):
        probability = random.randint(1, 10)
        if probability <= 3:
            index1 = random.randint(0, int(len(population) / 2))
        else:
            index1 = random.randint(int(len(population) / 2), len(population) - 1)
        probability = random.randint(1, 10)
        if probability <= 3:
            index2 = random.randint(0, int(len(population) / 2))
        else:
            index2 = random.randint(int(len(population) / 2), len(population) - 1)
        parent1 = population[index1]
        parent2 = population[index2]

        while True:
            index_a = random.randint(0, 31)
            index_b = random.randint(0, 31)
            index_c = random.randint(0, 31)

            parent1_a = parent1[0]
            parent1_b = parent1[1]
            parent1_c = parent1[2]

            parent2_a = parent2[0]
            parent2_b = parent2[1]
            parent2_c = parent2[2]

            child1_a = parent1_a[0:index_a] + parent2_a[index_a:len(parent1_a)]
            child1_b = parent1_b[0:index_b] + parent2_b[index_b:len(parent1_b)]
            child1_c = parent1_c[0:index_c] + parent2_c[index_c:len(parent1_c)]

            child2_a = parent2_a[0:index_a] + parent1_a[index_a:len(parent1_a)]
            child2_b = parent2_b[0:index_b] + parent1_b[index_b:len(parent1_b)]
            child2_c = parent2_c[0:index_c] + parent1_c[index_c:len(parent1_c)]

            child1 = [child1_a, child1_b, child1_c]
            child2 = [child2_a, child2_b, child2_c]

            if ((lower_bound <= float(bin_to_float(child1_a)) <= upper_bound)) and (
                    (lower_bound <= float(bin_to_float(child1_b)) <= upper_bound)) and (
                    (lower_bound <= float(bin_to_float(child1_c)) <= upper_bound)) and (
                    (lower_bound <= float(bin_to_float(child2_a)) <= upper_bound)) and (
                    (lower_bound <= float(bin_to_float(child2_b)) <= upper_bound)) and (
                    (lower_bound <= float(bin_to_float(child2_c)) <= upper_bound)):
                break

        children.append(child1)
        children.append(child2)
    for i in range(0, len(children)):
        population.append(children[i])


def selection():
    population.sort(key=fitness)
    size = int(len(population) / 2)
    for i in range(0, size):
        population.remove(population[0])


def fitness(individual_array):
    cost = 0
    a = float(bin_to_float(individual_array[0]))
    b = float(bin_to_float(individual_array[1]))
    c = float(bin_to_float(individual_array[2]))
    for i in range(0, len(x_train)):
        temp_x = x_train[i]
        temp_y = y_train[i]
        res_y = float(a * temp_x * temp_x + b * temp_x + c)
        cost += abs(res_y - temp_y)
    return -cost


def create_individual():
    return [float_to_bin(random.random() * (upper_bound - lower_bound) + lower_bound) for i in range(individual_size)]


def init_population():
    for i in range(0, genome_number):
        t = create_individual()
        population.append(t)


x_train = np.genfromtxt('x_train.csv', delimiter=',')
y_train = np.genfromtxt('y_train.csv', delimiter=',')
plt.plot(x_train, y_train, 'o', color='black')
plt.show()

generation_number = int(input("Please Enter Number Of Generations:"))
genome_number = int(input("Please Enter Number Of Genomes:"))
individual_size = int(input("Please Enter individual_size :"))
lower_bound = float(input("Please Enter Your Lower Bound:"))
upper_bound = float(input("Please Enter Your Upper Bound:"))

population = list()
init_population()

for i in range(0, generation_number):
    print('generation: %d' % (i))
    selection()
    crossover()
    mutation()
    population.sort(key=fitness, reverse=True)
    best_population = population[0]
    print (fitness(best_population))
    best_a = best_population[0]
    best_b = best_population[1]
    best_c = best_population[2]

    print(bin_to_float(best_a))
    print(bin_to_float(best_b))
    print(bin_to_float(best_c))
    print ('\n')

print ("final :")
population.sort(key=fitness, reverse=True)
best_population = population[0]
print (fitness(best_population))
best_a = best_population[0]
best_b = best_population[1]
best_c = best_population[2]

print(bin_to_float(best_a))
print(bin_to_float(best_b))
print(bin_to_float(best_c))
