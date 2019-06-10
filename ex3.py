from random import shuffle
import sys
import numpy as np

# def RelU(x)lambda x: max(0, x)


def shuffle_the_stuff(train_x, train_y):
    zipM = list(zip(train_x, train_y))
    shuffle(zipM)
    train_x, train_y = zip(*zipM)
    validation_size = int(len(train_x) * 0.2)
    train_x = train_x[validation_size:]
    validation_x = train_x[0: validation_size]
    train_y = train_y[validation_size:]
    validation_y = train_y[0:validation_size]
    return train_x, train_y, validation_x, validation_y


def loss(y, h1):
    return -(y * np.log(h1) + (1 - y) * np.log(1 - h1))


def forward(x, params):
    W1, b1 = [params[key] for key in ('W1', 'b1')]
    z1 = np.dot(W1, x) + b1
    # h1 = RelU(z1)
    # ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    # for key in params:
    #     ret[key] = params[key]
    return 0


def init_params():
    input_size = 28 * 28
    hidden_size = input_size / 2
    class_size = 10
    W1 = np.random.uniform(hidden_size, input_size)
    b1 = 0
    return {'W1': W1, 'b1': b1}


def NN(train_x, train_y):
    epchos = 10
    loss_sum = 0
    params = init_params()
    for i in range(epchos):
        for x, y in zip(train_x, train_y):
            output = forward(x, params)
            loss_sum += loss(y, output)



if __name__ == '__main__':
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2])
    # test_x = np.loadtxt(sys.argv[3])
    train_x, train_y, validation_x, validation_y = shuffle_the_stuff(train_x, train_y)
    NN(train_x, train_y)

    print(4)
