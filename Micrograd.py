import math
import random

class Value:

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other , Value) else Value(other)    # for operations with integer (like x + 1)
    out = Value(self.data + other .data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad 
      other.grad += 1.0 * out.grad

    out._backward = _backward
    return out
  
  def __radd__(self, other):    # when doing operations like 2 + x instead of x + 2
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    
    out._backward = _backward
    return out

  def __pow__(self, other):      # power operation needed for division
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self, ), f'**{other}')

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad

    self._backward = _backward
    return out

  def exp(self):          # Exponentiation
    out = Value(math.exp(self.data), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad

    self._backward = _backward
    return out


  def tanh(self):    # Activation function
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad

    out._backward = _backward
    return out

  def __rmul__(self, other):    # needed when doing operations like 2 * x instead of x * 2
    return self * other

  def __neg__(self):     # needed for subtracting and division
    return self * -1

  def __sub__(self, other):   # subtraction
    return self + (-other)

  def __truediv__(self, other):
    return self * (other**-1)
  

  def backward(self):   # Function for automatic backward propagation

    topo = []    # topological sort from end
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
