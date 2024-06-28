import numpy as np

if __name__ == '__main__':
    v1 = np.array(np.random.rand(3, 1250))
    v2 = np.array(np.random.rand(3, 5))
    v3 = np.array(np.random.rand(3, 1))

    r = np.concatenate((v1, v2, v3), axis=1)
    print(r.shape)