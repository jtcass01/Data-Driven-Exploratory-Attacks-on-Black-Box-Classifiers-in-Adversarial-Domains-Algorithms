import numpy as np
import matplotlib.pyplot as plt

from NeuralNetworkUtilities import initialize_random, propagate_forward, propagate_backward

class NeuralNetwork(object):
    def __init__(self, layers_dimensions):
        self.parameters = initialize_random(layers_dimensions)
        self.costs = []

    def train(self, X, Y, learning_rate = 0.0075, num_iterations = 30000):
        for epoch in range(0, num_iterations):
            AL, caches = propagate_forward(X, self.parameters)

            self.costs.append(self.compute_cost(AL, Y))

            gradients = propagate_backward(AL, Y, caches)

            self.parameters = self.update_parameters(self.parameters, gradients, learning_rate)

            if epoch % 1000 == 0:
                print(self.costs[len(self.costs)-1], 'cost @ epoch: ', epoch)

        self.plot_costs(self.costs, learning_rate)
        self.plot_decision_boundary(lambda x: self.predict_decision(x.T), X, Y)

    def plot_costs(self, costs, learning_rate):
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title('Learning rate =' + str(learning_rate))
        plt.show()

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(1, L + 1):
            parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
            parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]
        return parameters

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = -1 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(AL)))

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())

        return cost

    def predict_decision(self, X):
        """
        Used for plotting decision boundary.

        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (m, K)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """

        # Predict using forward propagation and a classification threshold of 0.5
        a3, cache = propagate_forward(X, self.parameters)
        predictions = (a3 > 0.5)

        return predictions

    def plot_decision_boundary(self, model, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
        plt.show()

def load_planar_testset():
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


if __name__ == '__main__':
    test_network = NeuralNetwork(layers_dimensions=[2,4,4,1])

    X, Y = load_planar_testset()

    print(X.shape, 'X')
    print(Y.shape, 'Y')

    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    test_network.train(X, Y)
