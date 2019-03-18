import sys  
import numpy as np
import random
from tqdm import tqdm_notebook

import pickle

def enumerate_grammar(t):
    grammar = set()
    if t.terminal:
        return(grammar)
    grammar.add(t.data)
    if t.left:
        grammar = grammar | enumerate_grammar(t.left)
    if t.right:
        grammar = grammar | enumerate_grammar(t.right)
    return(grammar)

def build_pcfg(training_trees, grammar, lexicon):
    grammar_pcfg = {k : dict() for k in grammar}
    lexicon_pcfg = {k : dict() for k in lexicon}
    # Fonction annexe pour parcourir les arbres
    def rec_on_tree(t): 
        
        if t.terminal:
            return
        
        for u in [t.data]+t.units:
            if u in grammar_pcfg:
                next_seq = ""
                for c in t.get_children():
                    if c.terminal:
                        lexicon_pcfg[c.data.lower()][u] = lexicon_pcfg[c.data.lower()].get(u,0) + 1
                        continue
                    next_seq += c.data + "-"
                    
                if len(next_seq) > 0:
                    grammar_pcfg[u][next_seq[:-1]] = grammar_pcfg[u].get(next_seq[:-1],0) + 1
                
        for c in t.get_children():
            rec_on_tree(c)
            
    for t in training_trees:
        rec_on_tree(t)
    
    for i in grammar_pcfg.keys():
        s = sum(grammar_pcfg[i].values())
        for j in grammar_pcfg[i].keys():
            grammar_pcfg[i][j] /= s
    
    for i in lexicon_pcfg.keys():
        s = sum(lexicon_pcfg[i].values())
        for j in lexicon_pcfg[i].keys():
            lexicon_pcfg[i][j] /= s
    return(grammar_pcfg,lexicon_pcfg)

    
def build_reverse_grammar_pcfg(grammar_pcfg):
    reverse_grammar_pcfg = dict()
    for k, children in grammar_pcfg.items():
        for child,p in children.items():
            if child not in reverse_grammar_pcfg:
                reverse_grammar_pcfg[child] = dict()
            reverse_grammar_pcfg[child][k] = p
    return(reverse_grammar_pcfg)

def save_pcfg(filename, grammar_pcfg, lexicon_pcfg):
    with open('%s_grammar.pickle'%filename, 'wb') as handle:
        pickle.dump(grammar_pcfg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s_lexicon.pickle'%filename, 'wb') as handle:
        pickle.dump(lexicon_pcfg, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pcfg(filename):
    with open('%s_grammar.pickle'%filename, 'rb') as handle:
        grammar_pcfg = pickle.load(handle)
    with open('%s_lexicon.pickle'%filename, 'rb') as handle:
        lexicon_pcfg = pickle.load(handle)
    return grammar_pcfg, lexicon_pcfg
