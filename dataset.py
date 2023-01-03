#!/usr/bin/env python3
import scipy.io
import numpy as np
import random

import decisiontree

random.seed(2)

wine_data = scipy.io.loadmat('./data/wine.mat')

X = wine_data['X']
y = wine_data['y'].ravel()

np.resize(X, (X.shape[0], X.shape[1] + 1))

X[:,-1] = y

attribute_names = [info[0] for info in wine_data["attributeNames"][0]]
class_names = [info[0][0] for info in wine_data["classNames"]]

# Shuffle data randomly to evenly distribute the classes among the records.
X_shuffled = np.zeros(X.shape)
shuffled_order = list(range(X.shape[0]))
random.shuffle(shuffled_order)
for old, new in enumerate(shuffled_order):
    X_shuffled[new,:] = X[old,:]

tree = decisiontree.DecisionTree(X_shuffled)

original_tree = tree.copy()

print(f'{X.shape[0]} datapoints')

f2 = tree.score()
print(f2)

# Target fitness is set to the maximum achieved fitness found before increasing
# the dataset_size. Thereafter it is lowered by 0.0001 per generation but always
# 0.9 at minimum.
target_fitness = 1.0

last_successful = None

for iteration in range(3000):
    copy = tree.copy()
    copy.mutate_leaf()
    f3 = copy.score()
    if f3 > f2:
        f2 = f3
        tree = copy
    if f2 >= target_fitness:
        last_successful = tree.copy()
        print(f'Target fitness {target_fitness:.5f} has been achieved on iteration {iteration}, dataset size has been increased.')
        print('by tree: with score ', f2, ' over ', tree.dataset_size, ' records')
        tree.increase_dataset_size()
        target_fitness = f2
        f2 = tree.score()
    else:
        target_fitness = max(0.9, target_fitness - 1e-4)


print('-- results --')
print(f2)
print(tree)
print(tree.score())
print(tree.dataset_size)

print('--')
print(last_successful)
print(last_successful.summary())
