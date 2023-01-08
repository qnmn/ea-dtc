import random

from typing import *

import numpy as np

from node import Node, to_btree


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
        self.test_data = data
        self.contraction_chance = 0.7
        if attribute_names is not None:
            self.attribute_names = attribute_names
        else: # Default names if none were given.
            self.attribute_names = [f'V{i + 1}' for i in range(self.test_data.shape[1])]

        if dataset_size <= 0:
            self.dataset_size = 2
        else:
            self.dataset_size = dataset_size
        self.max_depth = max_depth

        if class_attribute is None:
            self.class_attr = data.shape[1] - 1
        else:
            self.class_attr = class_attribute

        self.ranges = ranges
        if ranges is None:
            self.ranges = compute_ranges(self.test_data)

        self.class_names = class_names
        if class_names is None:
            lower, upper = self.ranges[:,self.class_attr]
            self.class_names = {i: f'C{i}' for i in range(int(lower), int(upper + 1))}

        self.root = root
        if root is None:
            predicted_class = random.choice(
                range(int(self.ranges[1, self.class_attr]) + 1))
            self.root = self.random_prediction()

    def copy(self):
        tree = DecisionTree(self.test_data,
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
        if self.root is None:
            return 0

        if data is None or data is self.test_data:
            data = self.test_data
            indices = np.ones(self.dataset_size, dtype=int)
            results = np.zeros(self.dataset_size)
            self.root.cache_decide(indices, results)

            correct = (results == self.test_data[:self.dataset_size,self.class_attr])
            score = correct.astype(int, copy=False).mean()
            return score
        else:
            num_entries = data.shape[0]
            return sum(
                self.decide(data[i, :]) == data[i, self.class_attr]
                for i in range(num_entries)) / num_entries

    def decide(self, entry):
        return self.root.decide(entry)

    def mutate_prediction(self):
        candidates = [
            node for node, depth in self.root.iterate_nodes()
            if depth < self.max_depth and node.is_pred
        ]
        if candidates:
            question = random.choice(candidates)
            question.split_leaf()
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
        self.dataset_size = min(self.test_data.shape[0],
                                self.dataset_size + increase_by)

    def summary(self):
        return f'''Max depth: {self.max_depth}
Dataset size: {self.dataset_size} (out of {self.test_data.shape[0]})
Score with constrained dataset: {self.score()}
Score with full dataset: {self.score(self.test_data)}'''

    def render(self):
        btree = to_btree(self.root)
        if btree is not None:
            btree.graphviz().render()

    def predictions(self):
        if self.root is None:
            return []
        for node in self.root.iterate_nodes():
            if node.is_pred:
                yield node

    def questions(self):
        if self.root is None:
            return []
        for node in self.root.iterate_nodes():
            if not node.is_pred:
                yield node

    def mutate_question(self):
        pass
