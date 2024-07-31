import unittest
from src.model import load_dataset, train_and_evaluate, get_dataset_statistics


class TestModel(unittest.TestCase):

    def test_load_dataset(self):
        datasets = ['iris', 'wine', 'breast_cancer']
        for dataset in datasets:
            data, target = load_dataset(dataset)
            self.assertGreater(len(data), 0)
            self.assertGreater(len(target), 0)

    def test_train_and_evaluate(self):
        accuracy, error = train_and_evaluate('iris', 0.8, 0.2)
        self.assertGreaterEqual(accuracy, 0.5)  # simple check to ensure model is somewhat effective

    def test_get_dataset_statistics(self):
        stats = get_dataset_statistics('iris')
        self.assertIn('mean', stats.index)


if __name__ == '__main__':
    unittest.main()
