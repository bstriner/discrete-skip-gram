
from discrete_skip_gram.dataset import DatasetFiles
from nltk.corpus import brown

def main():
    data = DatasetFiles(brown)
    data.summary()

if __name__ =="__main__":
    main()
