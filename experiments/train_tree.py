import sys
import os

sys.path.append(os.path.dirname(__file__))
import skipgram_tree_train
import skipgram_tree_el_train
import skipgram_tree_b_train

def main():
    skipgram_tree_train.main()
    skipgram_tree_el_train.main()
    skipgram_tree_b_train.main()

if __name__=="__main__":
    main()