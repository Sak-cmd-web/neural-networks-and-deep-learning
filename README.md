# Neural Networks and Deep Learning

This repository contains code from Michael Nielsen's book, Neural Networks and Deep Learning and this project is inspired by the Youtube channel 3Blue1Brown.

The original code was written for Python 2, so I updated network.py and network2.py to work with Python 3.11 and 3.12, making it easier to run on modern Python versions and for everyone to implement and make changes on their own. This repo doesn't change the original codes or add any new methods, just making it compatible for everyone's computer :)

## Project overview

An artificial neural network (ANN) is a collection of simple numerical units
that transforms inputs through connected layers. Here it learns to recognize
MNIST handwritten digits: each 28 × 28 grayscale image is flattened into 784
pixel values, and the network chooses one of 10 output neurons, representing
digits 0 through 9.

The input layer has 784 values. Hidden layers learn patterns such as strokes
and curves. The final layer has 10 activations; the largest is the predicted
digit. During feedforward, values move from input to output.
Backpropagation calculates how each weight and bias contributed to an
error. Stochastic gradient descent (SGD) applies those gradients to small
random mini-batches, gradually improving predictions.

## Setup

From the repository root, create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the required NumPy version using either tool:

```bash
uv pip install -r requirements.txt
# or
python -m pip install -r requirements.txt
```

The repository includes `data/mnist.pkl.gz`. It must be a readable gzip archive
containing 50,000 training examples, 10,000 validation examples, and 10,000
test examples.

## Files

- `src/mnist_loader.py` loads MNIST and turns each image into a `(784, 1)`
  vector. Training labels become `(10, 1)` one-hot vectors; validation and test
  labels remain integer digit classes.
- `src/network.py` is the introductory ANN: sigmoid activations, quadratic
  cost, backpropagation, and mini-batch SGD.
- `src/network2.py` keeps the same core mathematics and adds cross-entropy
  cost, L2 regularization, improved initialization, monitoring, and JSON
  save/load support.

## Training and testing

Run these commands from the repository root. An epoch is one full pass through
training data; mini-batch size is examples per update; `eta` is the learning
rate.

Train the introductory ANN. Its `correct / 10000` output is test accuracy:

```bash
PYTHONPATH=src python -c 'import mnist_loader, network; train, _, test = mnist_loader.load_data_wrapper(); net = network.Network([784, 30, 10]); net.SGD(train, 30, 10, 3.0, test_data=test)'
```

Train the improved ANN. Its monitored `accuracy / 10000` is test accuracy:

```bash
PYTHONPATH=src python -c 'import mnist_loader, network2; train, _, test = mnist_loader.load_data_wrapper(); net = network2.Network([784, 30, 10]); net.SGD(train, 30, 10, 0.5, lmbda=5.0, evaluation_data=test, monitor_evaluation_accuracy=True); net.save("network2-model.json")'
```

Run the focused regression tests:

```bash
python -m unittest discover -s tests -v
```

`network2.py` can save and load a trained model. A prediction on an MNIST input
is the index of the largest output activation:

```python
import mnist_loader, network2
_, _, test = mnist_loader.load_data_wrapper()
net = network2.load("network2-model.json")
predicted_digit = net.feedforward(test[0][0]).argmax()
```

`network.py` has no save/load feature, and there is no separate command-line
prediction program. The `test` variable comes from `mnist_loader.load_data_wrapper()`.

## Modernization and limitations

The in-scope code was updated from Python 2 syntax and APIs to Python 3:
`cPickle` is binary `pickle` loading with Python 2-compatible encoding,
`xrange` became `range`, print calls were updated, loader `zip` results are
materialized for shuffling and sizing, and the MNIST path is repository-relative.
The original NumPy feedforward, backpropagation, initialization, cost, and
regularization behavior is preserved.

`src/network3.py`, `src/conv.py`, and related figure scripts remain legacy code.
They depend on old Theano APIs and are outside this migration.

## License

MIT License

Copyright (c) 2012-2022 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
