from discrete_skip_gram.prior import Prior

def main():
    p = Prior(4,2,5)
    for i in range(10):
        print p.prior_samples(5)

if __name__=="__main__":
    main()