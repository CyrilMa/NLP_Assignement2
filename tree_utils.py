import sys  
import numpy as np
import random
from tqdm import tqdm_notebook

class Node(object):
    def __init__(self, data, parent, terminal): 
        self.data = data
        self.children = []
        self.parent = parent
        self.terminal = terminal
        
    def __str__(self):
        string = self.data+" "
        for c in self.children:
            if c.terminal: 
                string += c.data
            else:
                string += "(" + str(c) + ")"
        return(string)
    
    def __eq__(self, other):
        return(str(self) == str(other))

    def to_sentence(self):
        if self.terminal:
            return self.data
        
        sentence = ""
        for c in self.children:
            sentence += c.to_sentence() + " "
        return sentence[:-1]
    
    def get_children(self):
        return self.children
    
class BinaryNode(object):
    def __init__(self, data, terminal): 
        self.data = data
        self.left = None
        self.right = None
        self.terminal = terminal
        self.units = []
    
    def __str__(self):
        string = self.data+" "
        if self.left :
            if self.left.terminal:
                string+= self.left.data
            else:
                string+= "(" + str(self.left) +")"
                
        if self.right :
            if self.right.terminal:
                string+= self.right.data
            else:
                string+= "(" + str(self.right) +")"
        return(string)

    def __eq__(self, other):
        return(str(self).lower() == str(other).lower())

    def accuracy(self, other):
        pos_self, pos_other = [],[]
        def aux(t):
            if t.left.terminal:
                return([t.data])
            return aux(t.left) + aux(t.right)
        pos_self = aux(self)
        pos_other = aux(other)
        return(sum(int(s == o) for s,o in zip(pos_self,pos_other))/len(pos_self))

    def to_sentence(self):
        if self.right:
            return(self.left.to_sentence()+" "+self.right.to_sentence())
        return(self.left.data)
    
    def get_children(self):
        if self.right:
            return [self.left, self.right]
        if self.left:
            return([self.left])
        return([])

def to_non_binary(bin_node):
    def aux(t):
        if not t:
            return []
        if t.terminal:
            return [Node(t.data, None, True)]
        if len(t.data.split("-"))>1:
            return aux(t.left) + aux(t.right)
        
        node = Node(t.data, None, False)
        node.children = aux(t.left) + aux(t.right)
        
        return [node]
    return aux(bin_node)[0]

def to_binary(node): 
    # START STEP, DEL STEP already done by construction (we trust the data here)
    bin_node = BinaryNode(node.data, node.terminal)
    if node.terminal:
        return bin_node

    if len(node.children) == 1:
        if node.children[0].terminal:
            bin_node.left = to_binary(node.children[0])
            return(bin_node)
        # UNIT STEP
        bin2 = to_binary(node.children[0])
        bin2.units.append(bin2.data)
        bin2.data = bin_node.data
        return(bin2)

    # TERM STEP : unuseful in our case
#     for i in range(len(node.children)):
#         if node.children[i].terminal:
#             new_node = Node(node.children[i].data, self, False)
#             new_node.children.append(node.children[i])
#             node.children[i] = new_node

    if len(node.children) == 2:
        bin_node.left = to_binary(node.children[0])
        bin_node.right = to_binary(node.children[1])

    # BIN STEP
    if len(node.children) > 2 :
        bin_node.left = to_binary(node.children[0])
        right_node_lab = ""
        for c in node.children[1:]:
            right_node_lab += c.data + "-" 
        right_node = Node(right_node_lab[:-1], bin_node, False)
        right_node.children = node.children[1:]
        bin_node.right = to_binary(right_node)

    return(bin_node)

def build_trees(dataset):
    non_terminal_symbol = set()
    terminal_symbol = set()
    trees = []

    for sentence in dataset:
        temp = sentence.split()

        current_node = Node("START",None,False)
        non_terminal_symbol.add("START")
        trees.append(current_node)

        for i in range(len(temp)):
            if temp[i][0]=='(':
                word = temp[i][1:].split("-")[0] 
                new_node = Node(word, current_node, False)
                current_node.children.append(new_node)
                current_node = new_node
                non_terminal_symbol.add(word)
            else:
                j = 0
                while temp[i][-j-1]==')':
                    j += 1
                word = temp[i][:-j]

                new_node = Node(word, current_node, True)
                current_node.children.append(new_node)
                for i in range(j):
                    current_node = current_node.parent
                terminal_symbol.add(word.lower())
    return(trees, terminal_symbol, non_terminal_symbol)

def build_bin_trees(dataset_trees): 
    bin_trees = []
    for tree in dataset_trees:
        bin_trees.append(to_binary(tree))
    return(bin_trees)

