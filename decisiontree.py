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
                 ranges=None):
        self.data = data
        if dataset_size <= 0:
            self.dataset_size = 2
        else:
            self.dataset_size = dataset_size
        self.max_depth = max_depth
        if class_attribute is None:
            self.class_attr = data.shape[1] - 1
        else:
            self.class_attr = class_attribute
        self.root: Optional[Node] = root
        if ranges is None:
            self.ranges = compute_ranges(self.data)
        else:
            self.ranges = ranges

        if root is None:
            predicted_class = random.choice(
                range(int(self.ranges[1, self.class_attr]) + 1))
            self.root = self.random_prediction()

    def copy(self):
        tree = DecisionTree(self.data,
                            self.dataset_size,
                            self.class_attr,
                            root=None,
                            max_depth=self.max_depth,
                            ranges=self.ranges)
        tree.root = self.root.copy(tree)
        return tree

    def score(self, test_data=None):
        if self.root is None:
            return 0
        if test_data is None:
            test_data = self.data
            num_entries = self.dataset_size
        else:
            num_entries = test_data.shape[0]

        return sum(
            self.decide(test_data[i, :]) == test_data[i, self.class_attr]
            for i in range(num_entries)) / num_entries

    def decide(self, entry):
        return self.root.decide(entry)

    def mutate_leaf(self):
        candidates = [
            node for node, depth in self.root.iterate_nodes()
            if depth < self.max_depth
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
        self.dataset_size = min(self.data.shape[0],
                                self.dataset_size + increase_by)

    def summary(self):
        return f'''Max depth: {self.max_depth}
Dataset size: {self.dataset_size} (out of {self.data.shape[0]})
Score with constrained dataset: {self.score()}
Score with full dataset: {self.score(self.data)}'''

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
