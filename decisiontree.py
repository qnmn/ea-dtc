from collections import namedtuple
from enum import Enum, auto
import random

from typing import *

import binarytree as bt
import numpy as np


class Operator(Enum):
    LT = ('<')
    EQ = ('=')
    GT = ('>')

    def __init__(self, rep):
        self.rep = rep

def to_btree(node):
    if node is None:
        return None
    else:
        root = bt.Node(str(node))
        root.left = to_btree(node.left)
        root.right = to_btree(node.right)
        return root


class Node():

    def __init__(self, tree, var, val, is_pred=True, op=Operator.EQ):
        self.var = var
        self.is_pred = is_pred
        self.op = op
        self.val = val

        self.tree: DecisionTree = tree

        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

    def decide(self, entry):
        if self.is_pred:
            return self.val
        else:
            var_val = entry[self.var]
            if self.op == Operator.LT and var_val < self.val or self.op == Operator.EQ and var_val == self.val or self.op == Operator.GT and var_val > self.val:
                return self.left.decide(entry)
            else:
                return self.right.decide(entry)

    def split_leaf(self):
        self.var = random.choice([
            var for var in range(self.tree.data.shape[1])
            if var != self.tree.class_attr
        ])
        self.is_pred = False

        self.op = random.choice([Operator.LT,
                                 Operator.GT])  # TODO: Operator.EQ

        v_min, v_max = self.tree.ranges[:, self.var]
        self.val = v_min + random.random() * (v_max - v_min)

        # Create two child predictions.
        self.left = self.tree.random_prediction()
        self.right = self.tree.random_prediction()

    def mutate_leaf(self, depth_threshold):
        if depth_threshold <= 0:
            return False
        if self.is_pred:
            # TODO: Other mutations besides splitting.
            self.split_leaf()

            return True
        else:
            # TODO: This choice should be even among all leafs.
            if random.random() < 0.5:
                self.left.mutate_leaf(depth_threshold - 1)
            else:
                self.right.mutate_leaf(depth_threshold - 1)

    def __str__(self):
        if self.is_pred:
            return f'P: {self.val}'
        else:
            return f'Q: {self.var} {self.op.rep} {self.val:.3f}'

    def copy(self, new_root):
        new_node = Node(new_root, self.var, self.val, self.is_pred, self.op)
        if self.left is not None:
            new_node.left = self.left.copy(new_root)
        if self.right is not None:
            new_node.right = self.right.copy(new_root)
        return new_node

    def iterate_nodes(self):
        if self.left is not None:
            yield from self.left.iterate_nodes()
        yield self
        if self.right is not None:
            yield from self.right.iterate_nodes()


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
        return self.root.mutate_leaf(self.max_depth)

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
Dataset size: {self.dataset_size}
Score with constrained dataset: {self.score()}
Score with full dataset: {self.score(self.data)}'''
