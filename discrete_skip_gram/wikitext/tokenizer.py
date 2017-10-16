def tokenize(path):
    tokens = []
    for line in open(path, 'r'):
        line = line.rstrip()
        if len(line) > 0:
            words = line.split()
            for word in words:
                tokens.append(word)
            tokens.append('<eos>')
    return tokens
