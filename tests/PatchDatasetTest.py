import unittest
from dataset.PatchDataset import PatchDataset


class PatchDatasetTest(unittest.TestCase):

    def test_returned_patch_dimensions(self):
        dataset = PatchDataset('./test_data/test', (7, 7))
        first_tensor = dataset[0]
        expected_shape = first_tensor.shape
        lst = []
        for idx in range(len(dataset)):
            tensor = dataset[idx]
            if tensor.shape != expected_shape:
                lst.append([idx, tensor.shape])
        self.assertEqual(len(lst), 0)
