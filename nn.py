import math
import numpy as np
import random
from Micrograd import Value

class Neuron:
    """A single unit of neuron"""

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """Returns non-linear output"""
        out = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        act = out.tanh()
        return act
    
    def parameters(self):
        """Return parameters used in calculation of output (weight and bias terms)"""
        return self.w + [self.b]
    

class Layer:
    """ A single layer of Neurons"""

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out
    
    def parameters(self):
        """Returns paramters of a layer"""
        # params = []

        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params
        return [p for neuron in self.neurons for p in neuron.parameters()]
    

class MLP:
    """Multi Layer Neural Network"""

    # nouts is a list containing number of neurons in each layer
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))] 

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0]
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

n = MLP(3, [4, 4, 1]) # Dimensions of neural network

# Before training output
ypred = [n(x) for x in xs]
print(f'Desired Outcomes: {ys}')
print(f'Result without training: {ypred}')

## Gradient Descent
for i in range(20):

    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward pass
    for p in n.parameters():
        p.grad = 0
    loss.backward()

    # Update weights
    for p in n.parameters():
        p.data += -0.05 * p.grad
    
    print(i, loss.data)


# After training
ypred = [n(x) for x in xs]
print(f'Desired Outcomes: {ys}')
print(f'Result after training: {ypred}')