from enum import Enum
import random

import binarytree as bt
import numpy as np

SPLIT_CHANCE = 1/3
# Split 1.0 between VAR_chance VAL_chance and the chance to change the operator.
VAR_CHANCE = 1/3
VAL_CHANCE = 1/3

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

    def __init__(self, tree, var, val, is_pred=True, op=Operator.EQ, cache=None):
        self.var = var
        self.is_pred = is_pred
        self.op = op
        self.val = val

        self.tree = tree

        self.left = None
        self.right = None

        self.cache = cache

    def decide(self, entry):
        if self.is_pred:
            return self.val
        else:
            var_val = entry[self.var]
            if self.op == Operator.LT and var_val < self.val or self.op == Operator.EQ and var_val == self.val or self.op == Operator.GT and var_val > self.val:
                # Left branch is yes.
                return self.left.decide(entry)
            else:
                # Right branch is no.
                return self.right.decide(entry)

    def split_leaf(self):
        self.mutate()

    def mutate(self):
        if self.is_pred:
            if random.random() < SPLIT_CHANCE:
                # Split the prediction.
                self.var = random.choice([
                    var for var in range(self.tree.test_data.shape[1])
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
            else:
                # Change prediction
                lower, upper = self.tree.ranges[:,self.tree.class_attr]
                if upper - lower == 1:
                    self.val = lower if self.val == upper else upper
                else:
                    new_val = random.randint(lower, upper - 1)
                    if new_val >= self.val:
                        self.val = new_val + 1
                    else:
                        self.val = new_val
        else:
            # Mutate the question node.

            self.cache = None

            # If applicable: 1/4 chance of turning the question into a prediction.
            if self.left.is_pred and self.right.is_pred and random.random() < self.tree.contraction_chance:
                self.is_pred = True
                self.left = None
                self.right = None
                self.var = self.tree.class_attr
                self.op = Operator.EQ
                self.val = random.choice(
                    range(int(self.tree.ranges[1, self.tree.class_attr]) + 1))
            else:
                # Even split between changing variable, operator or value.
                r = random.random()
                if r < VAR_CHANCE:
                    # Change variable.
                    self.var = random.randrange(0, self.tree.test_data.shape[1] - 1)
                    if self.var >= self.tree.class_attr:
                        self.var += 1
                    # Select random value from range.
                    lower, upper = self.tree.ranges[:,self.var]
                    self.val = lower + random.random() * (upper - lower)
                elif r < VAR_CHANCE + VAL_CHANCE:
                    # Change value.
                    lower, upper = self.tree.ranges[:,self.var]
                    alpha = 0.1 * (upper - lower)
                    rand_lower = max(lower, self.val - alpha)
                    rand_upper = min(upper, self.val + alpha)
                    self.val = rand_lower + random.random() * (rand_upper - rand_lower)
                else:
                    # Change opeartor:
                    # FIXME: This only makes sense for int varriables.
                    # if self.op == Operator.EQ:
                    #     if r < 1/6:
                    #         self.op = Operator.LT
                    #     else:
                    #         self.op = Operator.GT
                    # else:
                    #     self.op = Operator.EQ

                    if self.op == Operator.LT:
                        self.op = Operator.GT
                    else:
                        self.op = Operator.LT


    def __str__(self):
        if self.is_pred:
            return self.tree.class_names[self.val]
        else:
            var_name = self.tree.attribute_names[self.var]
            return f'{var_name} \\{self.op.rep} {self.val:.3f}?'

    def copy(self, new_root):
        new_node = Node(new_root, self.var, self.val, self.is_pred, self.op, self.cache)
        if self.left is not None:
            new_node.left = self.left.copy(new_root)
        if self.right is not None:
            new_node.right = self.right.copy(new_root)
        return new_node

    def iterate_nodes(self, depth=1):
        if self.left is not None:
            yield from self.left.iterate_nodes(depth+1)
        yield (self, depth)
        if self.right is not None:
            yield from self.right.iterate_nodes(depth+1)

    def cache_decide(self, indices, results):
        if self.is_pred:
            results[indices] = self.val
        else:
            if self.cache is None:
                self.compute_cache()
            n = results.shape[0]
            # All entries remaining in indices which fullfil the node condition.
            left = np.logical_and(self.cache[:n], indices)
            # All the others
            right = np.logical_xor(left, indices)
            self.left.cache_decide(left, results)
            self.right.cache_decide(right, results)

    def compute_cache(self):
        if self.is_pred:
            # No cache necessary for prediction nodes.
            self.cache = None
        else:
            if self.op == Operator.EQ:
                self.cache = self.tree.test_data[:,self.var] == self.val
            elif self.op == Operator.LT:
                self.cache = self.tree.test_data[:,self.var] < self.val
            else: # Operator.GT
                self.cache = self.tree.test_data[:,self.var] > self.val

