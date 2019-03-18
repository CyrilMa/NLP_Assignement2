import sys  
import numpy as np
import random
from tqdm import tqdm_notebook

import pickle
import time
import re

from sklearn.metrics.pairwise import cosine_similarity

DIGITS = re.compile("[0-9]", re.UNICODE)

class CorrecterEmbedder(object):
    
    def __init__(self, vocabulary, file='model/polyglot-fr.pkl', mode = "embedder"): 
        self.words, self.embeddings = pickle.load(open(file, 'rb'), encoding='latin1')
        print("Embeddings shape is {}".format(self.embeddings.shape))
        
        self.words_id = {w:i for (i, w) in enumerate(self.words)}
        self.id_words = {i:w for (i, w) in enumerate(self.words)}
        self.mistakes = dict()
        self.vocabulary = vocabulary
        self.mode = mode

    def save(self, filename):
        with open("%s.pickle"%filename, 'wb') as handle:
            pickle.dump((self.words,self.embeddings), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def set_mode(self, mode):
        if mode == "embedder" or mode == "correcter":
            self.mode = mode
        
    def embedding_word(self, word, forced_correction = False):
        d = 1
        if word not in self.words:
            _,d,word = self.correct_word(word, forced_correction = forced_correction)
        if not word:
            return np.inf, np.inf, None
        if d <= 0:
            return -np.inf, word
        return list(self.embeddings[self.words_id[word]])

    def build_embedded_lexicon(self, training_lexicon, filename):
        training_embedded_lexicon = dict()
        for w in training_lexicon:
            v = self.embedding_word(w)
            if v[0] != np.inf:
                training_embedded_lexicon[w] = v
        if filename:
            with open('training_embedded_lexicon.pickle', 'wb') as handle:
                pickle.dump(training_embedded_lexicon, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def correct_word(self, word, max_explo = 2, forced_correction = False):
        results = self.explore_case(word)
        results.sort()
        index, w = results[0]
        if index != 1e12:
            return 0, index, w

        word = DIGITS.sub("#", word)
        results = self.explore_case(word)
        results.sort()
        index, w = results[0]
        if index != 1e12:
            return 0, index, w

        # On cherche parmi les erreurs déjà commises 
        if word.lower() in self.mistakes: 
            w = word.lower()
            ed,word_id,correct = self.mistakes[w]
            if w.upper() == word:
                return ed, word_id, correct
            if w.title() == word:
                return ed, word_id, correct
            return ed, word_id, correct

        # On explore les mots proches
        words_list = {word}
        for i in range(max_explo):
            words_list = self.generate_close_words(words_list)
            results = []
            for w in words_list:
                results += self.explore_case(w)
            results.sort()
            index, w = results[0]
            if index != 1e12:
                self.mistakes[word.lower()] = (i, index, w)
                return ((i+1), index, w)

        if self.mode == "embedder" or not forced_correction:
            return np.inf, np.inf, None

        # Si on a pas trouvé, on arrête et on compare avec tous les mots, c'est supposé arriver rarement
        print("WARNING : Computing every edit distance for word '%s'. This might take some time ..."%word)
        distances_dict = [(edit_distance(word.lower(), w.lower()), i, w) for w, i in self.words_id.items()]
        distances_dict.sort()
        self.mistakes[word.lower()] = distances_dict[0]
        return(distances_dict[0])
    
    def explore_case(self, word, default = 1e12):
        normal = (self.words_id.get(word, default), word)
        lower = (self.words_id.get(word.lower(), default), word.lower())
        upper = (self.words_id.get(word.upper(), default), word.upper())
        title = (self.words_id.get(word.title(), default), word.title())
        if self.mode == "correcter" and lower[1] in self.vocabulary:
            lower = (lower[0]-default, lower[1])
        return [normal, lower, upper, title]
    
    def generate_close_words(self, word_set):
        new_words = set()
        for w in word_set:
            for i in range(len(w)):
                for letter in "abcdefghijklmnopqrstuvwxyz":
                    new_words.add(w[:i]+letter+w[i+1:]) # Substition
                    new_words.add(w[:i]+letter+w[i:]) # Insertion
                new_words.add(w[:i]+w[i+1:]) # Deletion
        return new_words

class Lexicon(object):
    def __init__(self, filename): 
        with open(filename+'.pickle', 'rb') as handle:
            self.embedded_lexicon = pickle.load(handle)

        self.embeddings = list(self.embedded_lexicon.values())
        self.words_id = {w:i for i,w in enumerate(self.embedded_lexicon.keys())} 
        self.id_words = {i:w for i,w in enumerate(self.embedded_lexicon.keys())} 
        
    def l2_nearest(self, e, k = 5):
        """Sorts words according to their Euclidean distance.
           To use cosine distance, embeddings has to be normalized so that their l2 norm is 1."""

        similarities = cosine_similarity(np.array(e).reshape(1, -1),self.embeddings).T
        sorted_similarities = sorted(enumerate(similarities), key=lambda t : -t[1])
        return zip(*sorted_similarities[:k])
    
    def id_to_words(self,i):
        return self.id_words[i]
    
    def words_to_id(self,w):
        return self.words_id[w]

def save_embedded_lexicon(filename, embedded_lexicon):
    with open('%s.pickle'%filename, 'wb') as handle:
        pickle.dump(embedded_lexicon, handle, protocol=pickle.HIGHEST_PROTOCOL)

def edit_distance(x, y):
    historic = dict()
    def aux(x, y):
        if y == "":
            return len(x)
        if x == "":
            return len(y)
        if not (x[:-1], y) in historic:
            historic[(x[:-1], y)] = aux(x[:-1], y)
        if not (x, y[:-1]) in historic:
            historic[(x, y[:-1])] = aux(x, y[:-1])
        if not  (x[:-1], y[:-1])in historic:
            historic[(x[:-1], y[:-1])] = aux(x[:-1], y[:-1])
        return min([historic[(x[:-1], y)]+1, historic[(x, y[:-1])]+1, historic[(x[:-1], y[:-1])]+int(x[-1] != y[-1])])
    return aux(x,y)

def oov_solver(word, lexicon, correcter_embedder, verbose = False):
    e = correcter_embedder.embedding_word(word, forced_correction = True)
    if e[0] == np.inf:
        print("OOV word")
        return
    if e[0] == -np.inf:
        return e[1]
    indices, similarities = lexicon.l2_nearest(e, 1)
    
    if verbose:
        neighbors = [lexicon.id_to_words(idx) for idx in indices]
        for i, (word, distance) in enumerate(zip(neighbors, similarities)):
              print(i, '\t', word, '\t\t', distance)
    return (lexicon.id_to_words(indices[0]))

