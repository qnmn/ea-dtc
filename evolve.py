#!/usr/bin/env python3
import argparse
import csv
import random

import scipy.io
from sklearn.model_selection import train_test_split
import sklearn.tree
import numpy as np

import decisiontree

random.seed(0)

max_depth = 13

parser = argparse.ArgumentParser()
parser.add_argument('dataset',
                    nargs='?',
                    choices=['wine', 'glass'],
                    default='wine')
args = parser.parse_args()

if args.dataset == 'wine':
    #################
    ### Wine data ###
    #################
    wine_data = scipy.io.loadmat('./data/wine.mat')

    X = wine_data['X']
    y = wine_data['y'].ravel()

    np.resize(X, (X.shape[0], X.shape[1] + 1))

    X[:, -1] = y

    attribute_names = [info[0] for info in wine_data["attributeNames"][0]]
    class_names = {
        i: info[0][0]
        for i, info in enumerate(wine_data["classNames"])
    }
else:
    ##################
    ### Glass data ###
    ##################
    glass_data = np.genfromtxt('./data/glass.data', delimiter=',')
    X = glass_data[:, 1:]
    # Important to exclude the first attribute: 'ID'
    attribute_names = [
        'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class'
    ]
    class_names = dict(
        enumerate([
            None, 'Building float', 'Building non-float', 'Vehicle float',
            'Vehicle non-flat', 'Container', 'Tableware', 'Headlamp'
        ]))

#####################
### Preprocessing ###
#####################
X_train, X_test = train_test_split(X,
                                   test_size=0.3,
                                   random_state=2,
                                   shuffle=True,
                                   stratify=X[:, -1])


##################################
### Greedy algorithm reference ###
##################################
dtc = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth,
                                          random_state=2).fit(
                                              X_train[:, :-1], X_train[:, -1])
greedy_score = dtc.score(X_test[:, :-1], X_test[:, -1])


#################
### Algorithm ###
#################
initial_tree = decisiontree.DecisionTree(X_train,
                                         max_depth=max_depth,
                                         attribute_names=attribute_names,
                                         class_names=class_names)

MAJOR_MUTATIONS = 2
MINOR_MUTATIONS = 80


# Perform one set of mutations.
def mutate(initial_tree):
    tree = initial_tree.copy()
    for _ in range(MAJOR_MUTATIONS):
        tree.mutate_question()
    f2 = tree.score()
    for _ in range(MINOR_MUTATIONS):
        copy = tree.copy()
        copy.mutate_prediction()
        f3 = copy.score()
        if f3 > f2:
            tree = copy  # Accept mutation.
            f2 = f3
    return tree, f2


fitness_history = []
def evolve(tree):
    cooldown = 0

    # Target fitness is set to the maximum achieved fitness found before increasing
    # the dataset_size. Thereafter it is lowered by 0.0001 per generation but always
    # 0.7 at minimum.
    target = 1.0

    fitness = tree.score()
    generation_count = 0
    while tree.dataset_size < tree.training_data.shape[0]:
        generation_count += 1

        # The paper introduces a cooldown period after succesful mutations that
        # increases the chance of two predictions being combined for 100 iterations.
        if cooldown > 0:
            cooldown -= 1
            tree.contraction_chance = 0.6
        else:
            tree.contraction_chance = 0.3

        mutated, mutated_fitness = mutate(tree)
        if mutated_fitness > fitness:
            # Mutation did improve, accept mutation.
            tree = mutated
            fitness = mutated_fitness
        else:
            # Mutation did not improve, adjust target and do not accept the mutation.
            target = max(0.7, target - 1e-4)
        if fitness > target:
            # Target has been reached succesfully.
            cooldown = 100

            fitness_history.append((tree.dataset_size, fitness)) # For visualization purposes.

            if tree.dataset_size == tree.training_data.shape[0]:
                # Tree has evolved to the entire dataset, end algorithm.
                break

            tree.increase_dataset_size()
            print(f'Fitness target {target:.4f} met. Dataset now {tree.dataset_size}/{tree.training_data.shape[0]}')
            target = fitness
            fitness = tree.score()
    return tree, fitness, generation_count


solution, fitness, generation_count = evolve(initial_tree)

# Visualization purposes
with open('fitness_history.csv', 'w', newline='') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    for size, fitness in fitness_history:
        w.writerow([size, fitness])

try: # Try rendering the tree via graphviz.
    solution.render()
except:
    print('Failed to render tree to pdf')
print(solution.summary())
print(f'Total generations: {generation_count}')
print(f'Score on test data: {solution.score(X_test)}')
print(f'Greedy score: {greedy_score}')
