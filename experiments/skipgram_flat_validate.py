
from discrete_skip_gram.validation import run_flat_validation

if __name__ == '__main__':
    output_path = "output/skipgram_flat_validate.txt"
    input_path = "output/skipgram_flat-els"
    run_flat_validation(input_path=input_path,
                        output_path=output_path)


