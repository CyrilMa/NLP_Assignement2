import sys
import re

from cyk_utils import CYK
from tree_utils import *

# "python test.py (PREFIX OF THE MODEL) (TESTING SET NOT ANNOTATED) (TESTING SET ANNOTATED)"

cyk = CYK(filename="model/"+sys.argv[1])
cyk.correcter_embedder.set_mode("correcter")

dataset = []
for line in open('data/'+sys.argv[2],'r', encoding="utf-8"):
    dataset.append(line)

gt_dataset = []
for line in open('data/'+sys.argv[3],'r', encoding="utf-8"):
    gt_dataset.append(line)
    
gt_dataset_trees, _, _ = build_trees(gt_dataset)
gt_dataset_bin_trees = build_bin_trees(gt_dataset_trees)

correct, j, success = 0, 0, 0


parenthesis = re.compile("\)\(", re.UNICODE)

for sent, bin_tree in zip(dataset,gt_dataset_bin_trees):
    j+=1
    best, hist, verbose = cyk.grammar_hypothesis(sent)

    print("Sentence :")
    print(sent)
    print()

    if best:
        file = open('output/prediction.txt','a')
        print(parenthesis.sub(") (", "( (SENT"+str(to_non_binary(best))[5:]+"))\n"))
        file.write(parenthesis.sub(") (", "( (SENT"+str(to_non_binary(best))[5:]+"))\n"))
        file.close()
        print("{} - {} words - Done in {:.3f} s".format(bin_tree.accuracy(best), verbose[0], verbose[1]))
        correct += bin_tree.accuracy(best)
        success += 1
    else:
        file = open('output/prediction.txt','a')
        file.write("Echec\n")
        file.close()
        print("Echec - {} words - Done in {:.3f} s".format(verbose[0], verbose[1]))    
    print("-------------------------------------------------------")

print("Global Accuracy : {:.3f} - Success Rate : {:.3f} - Accuracy when Success : {:.3f}".format(correct/j, success/j, correct/success))