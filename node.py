from enum import Enum
import random

import binarytree as bt

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
        self.mutate()

    def mutate(self):
        if self.is_pred:
            # Split the prediction.
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
        else:
            # Mutate the question node.


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
                if r < 1/3:
                    # Change opeartor:
                    if self.op == Operator.EQ:
                        if r < 1/6:
                            self.op = Operator.LT
                        else:
                            self.op = Operator.GT
                    else:
                        self.op = Operator.EQ
                elif r < 2/3:
                    # Change variable.
                    self.var = random.randrange(0, self.tree.data.shape[1] - 1)
                    if self.var >= self.tree.class_attr:
                        self.var += 1
                    # Select random value from range.
                    lower, upper = self.tree.ranges[:,self.var]
                    self.val = lower + random.random() * (upper - lower)
                else:
                    # Change value.
                    lower, upper = self.tree.ranges[:,self.var]
                    alpha = 0.1 * (upper - lower)
                    rand_lower = max(lower, self.val - alpha)
                    rand_upper = min(upper, self.val + alpha)
                    self.val = rand_lower + random.random() * (rand_upper - rand_lower)


    def __str__(self):
        if self.is_pred:
            return f'{self.val}'
        else:
            return f'V{self.var} \\{self.op.rep} {self.val:.3f}?'

    def copy(self, new_root):
        new_node = Node(new_root, self.var, self.val, self.is_pred, self.op)
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
