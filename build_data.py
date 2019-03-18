import random
random.seed(448)
from tree_utils import *

dataset = []
for line in open('data/sequoia-corpus+fct.MRG_STRICT.txt','r', encoding="utf-8"):
    dataset.append(line[2:-2])

# random.shuffle(dataset)
N_DATA = len(dataset)

training_set, validation_set, testing_set = dataset[:(9*N_DATA)//10],dataset[(9*N_DATA)//10:],[]

with open('data/training_set.txt', 'w',  encoding="utf-8") as f:
    for item in training_set:
        f.write("%s\n" % item)

validation_trees, _, _ = build_trees(validation_set)
validation_bin_trees = build_bin_trees(validation_trees)
with open('data/raw_validation_set.txt', 'w',  encoding="utf-8") as f:
    for item in validation_bin_trees:
        f.write("%s\n" % item.to_sentence())

with open('data/validation_set_ground_truth.txt', 'w',  encoding="utf-8") as f:
    for item in validation_set:
        f.write("%s\n" % item)

# testing_trees, _, _ = build_trees(testing_set)
# testing_bin_trees = build_bin_trees(testing_trees)
# with open('data/raw_testing_set.txt', 'w',  encoding="utf-8") as f:
#     for item in testing_bin_trees:
#         f.write("%s\n" % item.to_sentence())

# with open('data/testing_set_ground_truth.txt', 'w',  encoding="utf-8") as f:
#     for item in validation_set:
#         f.write("%s\n" % item)