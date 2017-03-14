
from discrete_skip_gram.dataset import DatasetFiles
from nltk.corpus import shakespeare

def main():
    data = DatasetFiles(shakespeare)
    data.summary()

if __name__ =="__main__":
    main()
