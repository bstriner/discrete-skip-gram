import sys
import argparse
from discrete_skip_gram.wiki_model import WikiModel
from discrete_skip_gram.wiki_dataset import WikiDataset

#5339722 docs in dump

def main(argv):
    parser = argparse.ArgumentParser(description='Train a DQN to control hyperparameters.')
    parser.add_argument('input', action="store", help='directory containing wikiextractor JSON extracts')
    parser.add_argument('output', action="store", help='directory to store output')
    args = parser.parse_args(argv)

    input_path = args.input
    output_path = args.output
    wiki = WikiModel(input_path)
    dataset = WikiDataset(wiki, output_path)
    dataset.process()


if __name__ == "__main__":
    main(sys.argv[1:])
