import sys

import numpy as np
from tqdm import tqdm

from cyk_utils import CYK
from oov_utils import CorrecterEmbedder, Lexicon, save_embedded_lexicon
from pcfg_utils import build_pcfg, enumerate_grammar
from tree_utils import build_bin_trees, build_trees

# run "python train.py (ANNOTATED_TRAINING_SET) (PREFIX_OF_GENERATED_MODEL) (facultative - EMBEDDED_LEXICON)"

training_set = []
for line in open("data/"+sys.argv[1],'r', encoding="utf-8"):
    training_set.append(line)

# Build the trees first
print("Building the trees ...")
training_tree, training_lexicon, training_grammar = build_trees(training_set)
training_bin_tree = build_bin_trees(training_tree)
training_lexicon, training_grammar = list(training_lexicon), list(training_grammar)

# Build the PCFG
print("Building the PCFG ...")
training_chomsky_grammar = set()
for t in training_bin_tree:
    training_chomsky_grammar = training_chomsky_grammar | enumerate_grammar(t)
    
training_chomsky_grammar = list(training_chomsky_grammar)
grammar_pcfg, lexicon_pcfg = build_pcfg(training_bin_tree, training_chomsky_grammar, training_lexicon)

# Build the OOV module
print("Building the OOV module ...")
polyglot = CorrecterEmbedder(training_lexicon)

if len(sys.argv) < 4:
    training_embedded_lexicon = dict()
    polyglot.set_mode("embedder")

    for i,w in tqdm(list(enumerate(training_lexicon))):
        v = polyglot.embedding_word(w)
        if v[0] != np.inf:
            training_embedded_lexicon[w] = v

    save_embedded_lexicon("model/"+sys.argv[2]+"_embedded_lexicon", training_embedded_lexicon)
    training_embedded_lexicon = Lexicon("model/"+sys.argv[2]+"_embedded_lexicon")

else:
    training_embedded_lexicon = Lexicon("model/"+sys.argv[3]+"_embedded_lexicon")

# Finally build the CYK module
print("Building the CYK module ...")
polyglot.set_mode("correcter")
cyk = CYK(lexicon_pcfg, grammar_pcfg, training_embedded_lexicon, polyglot)
cyk.save_model("model/"+sys.argv[2])

print("Done !")
