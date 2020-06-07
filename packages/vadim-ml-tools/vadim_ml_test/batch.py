from vadim_ml.batch import *

if __name__ == '__main__':
    print(list(batch_n([1, 2])))
    print(list(batch_n([np.ones((1, 2)), np.ones((1, 3))])))