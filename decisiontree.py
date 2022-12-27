from collections import namedtuple
from enum import Enum, auto
import random

from typing import *


class Operator(Enum):
    LT = auto()
    EQ = auto()
    GT = auto()

class Node:

    def __init__(self, var, val, is_pred=True, op=Operator.EQ):
        self.var = var
        self.is_pred = is_pred
        self.op = op
        self.val = val

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

class DecisionTree:
    def __init__(self, root=None):
        self.root: Optional[Node] = root

    def score(self, x_test, y_test):
        if self.root is None:
            return None

        return sum(self.decide(x_test[i,:]) == y_test[i] for i in range(x_test.shape[0]))

    def decide(self, entry):
        return self.root.decide(entry)


def generate_random_tree(class_var, classes):
    predicted_class = random.choice(classes)
    return DecisionTree(Node(class_var, predicted_class))
