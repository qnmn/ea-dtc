import random

from typing import *

import numpy as np

from node import Node, to_btree


# Computes the ranges of all attributes in a dataset.
def compute_ranges(data):
    ranges = np.zeros((2, data.shape[1]))
    ranges[0, :] = np.min(data, axis=0)
    ranges[1, :] = np.max(data, axis=0)
    return ranges


class DecisionTree:

    def __init__(self,
                 data,
                 dataset_size=0,
                 class_attribute=None,
                 root=None,
                 max_depth=20,
                 ranges=None,
                 attribute_names=None,
                 class_names=None):
        self.training_data = data
        # The chance that two sibling predictions are merged into a single one.
        self.contraction_chance = 0.5
        
        # The names of the dataset attributes.
        self.attribute_names = attribute_names
        if attribute_names is None:
            self.attribute_names = [f'V{i + 1}' for i in range(self.training_data.shape[1])]

        # Start with a dataset size of at least 2.
        self.dataset_size = dataset_size
        if dataset_size <= 0:
            self.dataset_size = 2

        self.max_depth = max_depth

        # Which attribute contains the classification. Choose the last one by default.
        self.class_attr = class_attribute
        if class_attribute is None:
            self.class_attr = data.shape[1] - 1

        # All the attribute ranges.
        self.ranges = ranges
        if ranges is None:
            self.ranges = compute_ranges(self.training_data)

        # The names of the different class values.
        self.class_names = class_names
        if class_names is None:
            lower, upper = self.ranges[:,self.class_attr]
            self.class_names = {i: f'C{i}' for i in range(int(lower), int(upper + 1))}

        # The root of the tree. Start with a random prediction if empty.
        self.root = root
        if root is None:
            self.root = self.random_prediction()

    def copy(self):
        tree = DecisionTree(self.training_data,
                            self.dataset_size,
                            self.class_attr,
                            root=None,
                            max_depth=self.max_depth,
                            ranges=self.ranges,
                            attribute_names=self.attribute_names,
                            class_names=self.class_names)
        tree.root = self.root.copy(tree)
        return tree

    def score(self, data=None):
        # Computes the accuracy (fitness function).
        if self.root is None:
            return 0

        if data is None or data is self.training_data:
            # If we are scoring our training data, we can make use of a cache
            # system to save the answers. This drastically improves performance
            # as these data points will be scored many times over.
            data = self.training_data
            indices = np.ones(self.dataset_size, dtype=int)
            results = np.zeros(self.dataset_size)
            self.root.cache_decide(indices, results)

            # A bool vector of which records were predicted correctly.
            correct = (results == self.training_data[:self.dataset_size,self.class_attr])
            score = correct.astype(int, copy=False).mean()
            return score
        else:
            num_entries = data.shape[0]
            return sum(
                self.decide(data[i, :]) == data[i, self.class_attr]
                for i in range(num_entries)) / num_entries

    def decide(self, entry):
        # Classify a single entry.
        return self.root.decide(entry)

    def mutate_prediction(self):
        candidates = [
            node for node, depth in self.root.iterate_nodes()
            if depth < self.max_depth and node.is_pred
        ]
        if candidates:
            question = random.choice(candidates)
            question.mutate()
            return True
        else:
            return False

    def random_prediction(self):
        predicted_class = random.choice(
            range(int(self.ranges[1, self.class_attr]) + 1))
        return Node(self, self.class_attr, predicted_class)

    def __str__(self):
        return str(to_btree(self.root))

    def increase_dataset_size(self):
        increase_by = int(1 + 0.02 * self.dataset_size)
        self.dataset_size = min(self.training_data.shape[0],
                                self.dataset_size + increase_by)

    def summary(self):
        return f'''Max depth: {self.max_depth}
Dataset size: {self.dataset_size} (out of {self.training_data.shape[0]})
Score with constrained dataset: {self.score()}
Score with full dataset: {self.score(self.training_data)}'''

    def render(self):
        # Use the binarytree library to visualize the tree.
        btree = to_btree(self.root)
        if btree is not None:
            btree.graphviz().render()

    def predictions(self):
        # List all predictions.
        if self.root is None:
            return []
        for node in self.root.iterate_nodes():
            if node.is_pred:
                yield node

    def questions(self):
        # List all questions.
        if self.root is None:
            return []
        for node in self.root.iterate_nodes():
            if not node.is_pred:
                yield node

    def mutate_question(self):
        candidates = [
            node for node, depth in self.root.iterate_nodes()
            if not node.is_pred
        ]
        if candidates:
            question = random.choice(candidates)
            question.mutate()
            return True
        else:
            return False
