import numpy as np
from typing import List

class Layer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def update(self, *args, **kwargs):
        pass  # If a layer has no parameters, then this function does nothing

class FeedForwardNetwork:
    
    def __init__(self, layers: List[Layer]):
        self.layers = layers
    
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._inputs = []
        for layer in self.layers:
            if train:
                self._inputs.append(x)
            x = layer.forward(x)
        return x
    
    def backward(self, dy: np.ndarray) -> np.ndarray:
        a = self._inputs[::-1]
        b = self.layers[::-1]
        c = zip(a, b)
        for x, lay in c:
            dy = lay.backward(x, dy)
        return dy
    
    def update(self, *args, **kwargs):
        for layer in self.layers:
            layer.update(*args, **kwargs)

class Linear(Layer):
    
    def __init__(self, insize: int, outsize: int) -> None:
        bound = np.sqrt(6. / insize)
        self.weight = np.random.uniform(-bound, bound, (insize, outsize))
        self.bias = np.zeros((outsize,))
        
        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)
   
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        self.dweight = np.dot(x.T, dy)
        self.dbias = np.sum(dy, axis=0)
        gradient = np.dot(dy, self.weight.T)
        return gradient
    
    def update(self, mode='SGD', lr=0.001, mu=0.9):
        if mode == 'SGD':
            self.weight -= lr * self.dweight
            self.bias -= lr * self.dbias
        else:
            raise ValueError('mode should be SGD, not ' + str(mode))

class ReLU(Layer):
    
    def __init__(self) -> None:
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        return dy * (x > 0)

class CrossEntropy:
    
    def __init__(self):
        pass
    
    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps,axis = 1).reshape(-1,1)

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        self.y = y
        self.t = t
        self.p = self.softmax(y)
        return -np.mean(np.log(self.p[np.arange(len(t)), t]))
    
    def backward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.p = self.softmax(y)
        self.p[np.arange(len(t)), t] -= 1
        return self.p / len(t)



def mlp_predict(number_of_attr, hidden_units, batch_size, ephocs_no, train_data, train_res, test_data):
    optimize_args = {'mode': 'SGD', 'lr': .005}
    net = FeedForwardNetwork([Linear(number_of_attr, hidden_units),
                          ReLU(),
                          Linear(hidden_units, 2)])
    cost_function = CrossEntropy()
    for ephocs in range(ephocs_no):
        for b_no, idx in enumerate(range(0, len(train_data), batch_size)):
            x = train_data[idx: idx + batch_size, :]
            train = train_res[idx: idx + batch_size]
            y = net.forward(x)
            loss = cost_function.forward(y, train)
            dy = cost_function.backward(y, train)
            net.backward(dy)
            net.update(**optimize_args)
    y = net.forward(test_data, train=False)
    return y
    pass