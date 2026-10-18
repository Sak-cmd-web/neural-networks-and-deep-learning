"""Focused regression tests for the Python 3 NumPy ANN implementations."""

import gzip
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mnist_loader
import network
import network2


def mnist_archive_is_readable():
    try:
        with gzip.open(ROOT / "data" / "mnist.pkl.gz", "rb") as archive:
            archive.read(1)
    except (OSError, EOFError):
        return False
    return True


class MnistLoaderTests(unittest.TestCase):
    @unittest.skipUnless(mnist_archive_is_readable(), "MNIST archive is unavailable or corrupt")
    def test_loads_expected_mnist_shapes(self):
        training, validation, testing = mnist_loader.load_data_wrapper()
        self.assertEqual((len(training), len(validation), len(testing)), (50000, 10000, 10000))
        self.assertEqual(training[0][0].shape, (784, 1))
        self.assertEqual(training[0][1].shape, (10, 1))
        self.assertEqual(validation[0][0].shape, (784, 1))
        self.assertIsInstance(validation[0][1], np.integer)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.2], [0.7]])
        self.y = np.array([[1.0], [0.0]])

    def test_basic_network_feedforward_backprop_and_update(self):
        net = network.Network([2, 3, 2])
        self.assertEqual(net.feedforward(self.x).shape, (2, 1))
        nabla_b, nabla_w = net.backprop(self.x, self.y)
        self.assertEqual([gradient.shape for gradient in nabla_b], [(3, 1), (2, 1)])
        self.assertEqual([gradient.shape for gradient in nabla_w], [(3, 2), (2, 3)])
        original_weights = [weight.copy() for weight in net.weights]
        original_biases = [bias.copy() for bias in net.biases]
        net.update_mini_batch([(self.x, self.y)], eta=0.5)
        self.assertTrue(any(not np.array_equal(before, after)
                            for before, after in zip(original_weights, net.weights)))
        self.assertTrue(any(not np.array_equal(before, after)
                            for before, after in zip(original_biases, net.biases)))

    def test_improved_network_serialization(self):
        net = network2.Network([2, 3, 2])
        net.update_mini_batch([(self.x, self.y)], eta=0.5, lmbda=0.0, n=1)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "model.json"
            net.save(filename)
            restored = network2.load(filename)
        self.assertEqual(restored.sizes, net.sizes)
        self.assertIs(restored.cost, network2.CrossEntropyCost)
        for expected, actual in zip(net.weights + net.biases,
                                    restored.weights + restored.biases):
            np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
