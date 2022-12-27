#!/usr/bin/env python3
import scipy.io

import decisiontree

wine_data = scipy.io.loadmat('./data/wine.mat')

X = wine_data['X']
y = wine_data['y'].ravel()

attribute_names = [info[0] for info in wine_data["attributeNames"][0]]
class_names = [info[0][0] for info in wine_data["classNames"]]

tree = decisiontree.generate_random_tree(None, [0, 1])

print(tree.score(X, y))
