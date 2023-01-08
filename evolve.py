#!/usr/bin/env python3
import random

import scipy.io
from sklearn.model_selection import train_test_split
import sklearn.tree
import numpy as np

import decisiontree

random.seed(0)

max_depth = 7

#################
### Wine data ###
#################
wine_data = scipy.io.loadmat('./data/wine.mat')

X = wine_data['X']
y = wine_data['y'].ravel()

np.resize(X, (X.shape[0], X.shape[1] + 1))

X[:,-1] = y

attribute_names = [info[0] for info in wine_data["attributeNames"][0]]
class_names = {i: info[0][0] for i, info in enumerate(wine_data["classNames"])}

##################
### Glass data ###
##################
# glass_data = np.genfromtxt('./data/glass.data', delimiter=',')
# X = glass_data[:,1:]
# # Important to exclude the first attribute: 'ID'
# attribute_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
# class_names = dict(enumerate([None, 'Building float', 'Building non-float', 'Vehicle float', 'Vehicle non-flat', 'Container', 'Tableware', 'Headlamp']))


#####################
### Preprocessing ###
#####################
X_train, X_test = train_test_split(
    X, test_size=0.3,
    random_state=2,
    shuffle=True,
    stratify=X[:,-1]
)


##################################
### Greedy algorithm reference ###
##################################

dtc = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth).fit(X_train[:,:-1], X_train[:,-1])
greedy_score = dtc.score(X_test[:,:-1], X_test[:,-1])


#################
### Algorithm ###
#################
initial_tree = decisiontree.DecisionTree(X_train, max_depth=max_depth, attribute_names=attribute_names, class_names=class_names)

MAJOR_MUTATIONS = 2
MINOR_MUTATIONS = 100

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
    while tree.dataset_size < tree.test_data.shape[0]:
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
            if tree.dataset_size == tree.test_data.shape[0]:
                break
            tree.increase_dataset_size()
            print(f'Fitness target {target:.4f} met. Dataset now {tree.dataset_size}/{tree.test_data.shape[0]}')
            target = fitness
            fitness = tree.score()
    return tree, fitness, generation_count


solution, fitness, generation_count = evolve(initial_tree)

solution.render()
print(solution.summary())
print(f'Total generations: {generation_count}')
print(f'Score on test data: {solution.score(X_test)}')
print(f'Greedy score: {greedy_score}')
