import time
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    logistic regression classifier
    """
    def __init__(self, maxiter, learning_rate):
        """
        initilize classifier
        :param maxiter: int, maximum number of iterations
        :param learning_rate: float, learning rate
        """
        self.maxiter = maxiter
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(x):
        """
        sigmoid(x) = 1 / (1 + exp(-x))
        :param x: float, a scalar
        :return: sigmoid(x)
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def predict(X, w):
        """
        predict the label of data points
        :param X: (n, d) ndarray, feature matrix of all data points
        :param w: (d, ) ndarray, weight vector
        :return: predicted labels of all data points
        """
        wx = np.dot(X, w)
        return [1 if i > 0 else 0 for i in wx]

    @staticmethod
    def accuracy(yhat, y):
        """
        calculate accuracy between predicted labels and true labels
        :param yhat: (ntest, ) ndarray, predicted labels
        :param y: (ntest, ) ndarray, true labels
        :return: accuracy between predicted labels and true labels
        """
        return sum(yhat == y) / y.shape[0]

    def log_likelihood(self, X, y, w):
        """
        calculate log likelihood
        :param X: (ntrain, d) ndarray, feature matrix of training data
        :param y: (ntrain, ) ndarray, labels of training data
        :param w: (d, ) ndarray, weight vector
        :return: sum of log likelihood over all data points
        """
        wx = np.dot(X, w)
        pos = y * np.log(self.sigmoid(wx))
        neg = (1 - y) * np.log(1 - self.sigmoid(wx))
        return sum(pos + neg)

    def fit(self, Xtrain, Ytrain, Xtest, Ytest):
        """
        function to train model
        :param Xtrain: (ntrain, d) ndarray, feature matrix of training data
        :param Ytrain: (ntrain, ) ndarray, labels of training data
        :param Xtest: (ntest, d) ndarray, feature matrix of test data
        :param Ytest: (ntest, ) ndarray, labels of test data
        :return: a list of test accuracies and a list of log likelihood after each iteration of training
        """
        # initialization
        ll = list()
        test_acc = list()
        w = np.zeros(Xtrain.shape[1])

        # training
        for _ in range(self.maxiter):
            # evaluate test accuracy and log likelihood
            test_acc.append(self.accuracy(self.predict(Xtest, w), Ytest))
            ll.append(-self.log_likelihood(Xtrain, Ytrain, w) / Xtrain.shape[0])

            # gradient ascent
            wx = np.dot(Xtrain, w)
            prob = self.sigmoid(wx)
            gradient = np.dot(Xtrain.T, Ytrain - prob) / Xtrain.shape[0]
            w += self.learning_rate * gradient

        # evaluate test accuracy and log likelihood after training
        test_acc.append(self.accuracy(self.predict(Xtest, w), Ytest))
        ll.append(-self.log_likelihood(Xtrain, Ytrain, w) / Xtrain.shape[0])
        return test_acc, ll

def draw_figure(lst, ylabel):
    """
    draw the graph of {ylabel} vs. number of iterations
    save it in a file named {ylabel}.png
    :param lst: list, list of data to be drawn on figure
    :param ylabel: string, lable of y-axis
    """
    plt.figure()  # initialize figure
    plt.plot(lst)  # plot
    plt.title('{} vs. number of iterations'.format(ylabel))  # set figure title
    plt.xlabel('number of iterations')  # set label on x-axis
    plt.ylabel(ylabel)  # set label on y-axis
    plt.tight_layout()
    plt.savefig('{}.png'.format(ylabel.replace(' ', '_')))  # save figure


if __name__ == '__main__':
    # load data
    data = scio.loadmat('MNIST_data/train_data.mat')
    Xtrain, Ytrain = data['X'], data['Y']
    Ytrain = Ytrain.reshape(-1)

    data = scio.loadmat('MNIST_data/test_data.mat')
    Xtest, Ytest = data['X'], data['Y']
    Ytest = Ytest.reshape(-1)

    # initialize classifier
    clf = LogisticRegression(maxiter=100,
                             learning_rate=0.1)

    # train and count total elapsed time for training
    start = time.time()
    test_acc, ll = clf.fit(Xtrain, Ytrain, Xtest, Ytest)
    end = time.time()
    print('elapsed time: {}s'.format(end-start))

    # draw figures
    draw_figure(test_acc, 'test accuracy')
    draw_figure(ll, 'training loss')