from discrete_skip_gram.wikitext.preprocessor import preprocess


def main():
    output_path = 'output/corpus'
    data_path = '../../../data/wikitext-2'
    preprocess(data_path=data_path, output_path=output_path)


if __name__ == '__main__':
    main()
