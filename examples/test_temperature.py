import numpy as np

def test():
    k = 5
    r = np.random.random((k,))
    p = np.exp(r)/np.sum(np.exp(r))
    print "P: {}".format(p)
    for temp in [0.1,0.5,1,1.5,20]:
        t = np.log(p)/temp
        pt = np.exp(t)/np.sum(np.exp(t))
        print "PT: {}, {}".format(temp, pt)



def main():
    for i in range(5):
        test()

if __name__ == "__main__":
    main()