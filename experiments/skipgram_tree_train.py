from discrete_skip_gram.tree_train import train_tree_battery


def main():
    epochs = 10
    iters = 3
    batches = 4096
    z_k = 2
    z_depth = 10
    outputpath = "output/skipgram_tree"
    betas = [0.85, 1.2]
    train_tree_battery(
        betas=betas,
        epochs=epochs,
        iters=iters,
        batches=batches,
        z_k=z_k,
        z_depth=z_depth,
        outputpath=outputpath)


if __name__ == "__main__":
    main()
