#!/usr/bin/env python3
import scipy.io
import numpy as np
import random

import decisiontree

random.seed(0)

#################
### Wine data ###
#################
wine_data = scipy.io.loadmat('./data/wine.mat')

X = wine_data['X']
y = wine_data['y'].ravel()

np.resize(X, (X.shape[0], X.shape[1] + 1))

X[:,-1] = y

attribute_names = [info[0] for info in wine_data["attributeNames"][0]]
class_names = [info[0][0] for info in wine_data["classNames"]]

##################
### Glass data ###
##################
glass_data = np.genfromtxt('./data/glass.data', delimiter=',')
X = glass_data[:,1:]


#####################
### Preprocessing ###
#####################

# Shuffle data randomly to evenly distribute the classes among the records.
X_shuffled = np.zeros(X.shape)
shuffled_order = list(range(X.shape[0]))
random.shuffle(shuffled_order)
for old, new in enumerate(shuffled_order):
    X_shuffled[new,:] = X[old,:]



#################
### Algorithm ###
#################
initial_tree = decisiontree.DecisionTree(X_shuffled, max_depth=13)

X = 2
Y = 80

def mutate(initial_tree):
    tree = initial_tree.copy()
    for _ in range(X):
        tree.mutate_question()
    f2 = tree.score()
    for _ in range(Y):
        copy = tree.copy()
        copy.mutate_prediction()
        f3 = copy.score()
        if f3 > f2:
            tree = copy # Accept mutation.
            f2 = f3
    return tree, f2

def evolve(tree):
    c1 = 0 # The number of generations for which tree contraction should be increased.
    c2 = 0
    # Target fitness is set to the maximum achieved fitness found before increasing
    # the dataset_size. Thereafter it is lowered by 0.0001 per generation but always
    # 0.9 at minimum.
    target = 1.0
    fitness = tree.score()
    generation_count = 0
    while tree.dataset_size < tree.data.shape[0]:
        if c1 > 0:
            c1 -= 1
            tree.contraction_chance = 0.6
        else:
            tree.contraction_chance = 0.3
        mutated, mutated_fitness = mutate(tree)
        generation_count += 1
        if mutated_fitness > fitness:
            tree = mutated
            fitness = mutated_fitness
        else:
            # TODO: Increment 'C2'
            target = max(0.7, target - 1e-4)
        if fitness > target:
            c1 = 100
            if tree.dataset_size == tree.data.shape[0]:
                break
            tree.increase_dataset_size()
            print(f'Fitness target {target:.4f} met. Dataset now {tree.dataset_size}/{tree.data.shape[0]}')
            target = fitness
            fitness = tree.score()
    return tree, fitness, generation_count


solution, fitness, generation_count = evolve(initial_tree)

solution.render()
print(solution.summary())
print(f'Total generations: {generation_count}')
