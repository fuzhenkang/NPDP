import unittest

import torch

from npdp_model_utils import compute_npdp_loss, last_valid_token_indices


class NPDPModelUtilsTest(unittest.TestCase):
    def test_last_valid_token_indices_supports_left_and_right_padding(self):
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        indices = last_valid_token_indices(None, mask, None, 4, mask.device)
        self.assertEqual(indices.tolist(), [1, 3])

    def test_mse_uses_activated_predictions(self):
        predictions = torch.tensor([[0.25], [0.75]])
        raw_logits = torch.tensor([[-1.0], [1.0]])
        labels = torch.tensor([0.0, 1.0])
        loss = compute_npdp_loss("mse", predictions, raw_logits, labels)
        self.assertTrue(torch.isclose(loss, torch.tensor(0.0625)))


if __name__ == "__main__":
    unittest.main()
