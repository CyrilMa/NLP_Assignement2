import sys  
import numpy as np
import random
from tqdm import tqdm_notebook

from itertools import product

from pcfg_utils import *
from tree_utils import *
from oov_utils import *

class CYK(object):
    
    def __init__(self, lexicon_pcfg=None, grammar_pcfg=None, lexicon=None, correcter_embedder=None, filename=None, MAX_SIZE=None):
        self.x_list = []
        self.y_list = []
        self.MAX_SIZE = MAX_SIZE

        self.training_lexicon = lexicon
        self.correcter_embedder = correcter_embedder
        self.lexicon_pcfg = lexicon_pcfg
        self.grammar_pcfg = grammar_pcfg

        if filename:
            self.load_model(filename)

        if lexicon:
            self.training_lexicon = lexicon
        
        if correcter_embedder:
            self.correcter_embedder = correcter_embedder
        
        if lexicon_pcfg:
            self.lexicon_pcfg = lexicon_pcfg

        if grammar_pcfg:
            self.grammar_pcfg = grammar_pcfg

        self.vocabulary = self.lexicon_pcfg.keys() 

        self.reverse_grammar_pcfg = dict()
        for k, children in self.grammar_pcfg.items():
            for child,p in children.items():
                if child not in self.reverse_grammar_pcfg:
                    self.reverse_grammar_pcfg[child] = dict()
                self.reverse_grammar_pcfg[child][k] = p

    def load_model(self, filename):
        self.grammar_pcfg, self.lexicon_pcfg = load_pcfg(filename)
        self.training_lexicon = Lexicon("%s_embedded_lexicon"%filename)
        try:
            self.correcter_embedder = CorrecterEmbedder(self.lexicon_pcfg.keys(),"%s_correcter_embedder"%filename)
        except:
            self.correcter_embedder = CorrecterEmbedder(self.lexicon_pcfg.keys())

    def save_model(self, filename):
        save_pcfg(filename, self.grammar_pcfg, self.lexicon_pcfg)
        save_embedded_lexicon("%s_embedded_lexicon"%filename, self.training_lexicon.embedded_lexicon)
        self.correcter_embedder.save("%s_correcter_embedder"%filename)

        
    def reset(self, sent):
        self.x_list = []
        self.y_list = []
        
        for w in sent:
            correct_w = w
            if w.lower() not in self.vocabulary:
                correct_w = oov_solver(w, self.training_lexicon, self.correcter_embedder)
            term_node = BinaryNode(w,True)
            pcfg = dict()
            for k,v in self.lexicon_pcfg[correct_w.lower()].items():
                new_node = BinaryNode(k,False)
                new_node.left = term_node
                pcfg[k] = (-np.log(v),new_node)
            self.x_list.append([pcfg])
            self.y_list.append([pcfg])

    def step(self,i,j):
        self.x_list[i].append(dict())
        self.y_list[j].append(dict())
        return(zip(self.x_list[i][:-1], self.y_list[j][:-1][::-1]))
    
    def add_e(self,e,i,j):
        if e[1].data in self.x_list[i][-1].keys():
            p,_ = self.x_list[i][-1][e[1].data]
            if p > e[0]:
                self.x_list[i][-1][e[1].data] = e
        else:
            self.x_list[i][-1][e[1].data] = e
            
        if e[1].data in self.y_list[j][-1].keys():
            p,_ = self.y_list[j][-1][e[1].data]
            if p > e[0]:
                self.y_list[j][-1][e[1].data] = e
        else:
            self.y_list[j][-1][e[1].data] = e

    def grammar_hypothesis(self, sent):
        s = time.time()
        sentence_split = sent.split()
        self.reset(sentence_split)
        m = len(sentence_split)

        for d in range(1, m): # For each diagonal
            for j in range(0, m-d): # For each case of the diagonal
                for set_left,set_right in self.step(j,j+d):
                    for (p_left,left), (p_right,right) in product(list(set_left.values()),list(set_right.values())):
                        if left.data+"-"+right.data in self.reverse_grammar_pcfg.keys():
                            for hyp, prob in self.reverse_grammar_pcfg[left.data+"-"+right.data].items():
                                new_node = BinaryNode(hyp, False)
                                new_node.left = left
                                new_node.right = right
                                self.add_e((p_left+p_right-np.log(prob), new_node), j, j+d)
                                
        return(self.x_list[0][-1].get("START", [None,None])[1], self.x_list, (m, time.time()-s))
