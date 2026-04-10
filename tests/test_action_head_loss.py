import unittest
from types import SimpleNamespace

import torch

from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead


class FlowmatchingActionHeadLossTest(unittest.TestCase):
    def _make_head(self, loss: str = "original", loc_loss_weight: float = 3.0, action_dim: int = 18):
        head = FlowmatchingActionHead.__new__(FlowmatchingActionHead)
        head.config = SimpleNamespace(
            loss=loss,
            loc_loss_weight=loc_loss_weight,
            action_dim=action_dim,
        )
        return head

    def test_original_loss_matches_plain_masked_mse(self):
        head = self._make_head(loss="original")
        pred_actions = torch.zeros((1, 1, 18), dtype=torch.float32)
        velocity = torch.ones((1, 1, 18), dtype=torch.float32)
        action_mask = torch.ones((1, 1, 18), dtype=torch.float32)

        loss = head._compute_loss(pred_actions, velocity, action_mask)

        self.assertAlmostEqual(loss.item(), 1.0)

    def test_loc_weighted_loss_emphasizes_first_three_dims(self):
        head = self._make_head(loss="loc_weighted", loc_loss_weight=3.0)
        pred_actions = torch.zeros((1, 1, 18), dtype=torch.float32)
        velocity = torch.ones((1, 1, 18), dtype=torch.float32)
        action_mask = torch.ones((1, 1, 18), dtype=torch.float32)

        loss = head._compute_loss(pred_actions, velocity, action_mask)

        self.assertAlmostEqual(loss.item(), 24.0 / 18.0)

    def test_loc_weighted_keeps_non_location_dims_unchanged(self):
        head = self._make_head(loss="loc_weighted", loc_loss_weight=3.0)
        pred_actions = torch.zeros((1, 1, 18), dtype=torch.float32)
        velocity = torch.zeros((1, 1, 18), dtype=torch.float32)
        velocity[:, :, 3:] = 1.0
        action_mask = torch.ones((1, 1, 18), dtype=torch.float32)

        loss = head._compute_loss(pred_actions, velocity, action_mask)

        self.assertAlmostEqual(loss.item(), 15.0 / 18.0)

    def test_masked_entries_do_not_contribute(self):
        head = self._make_head(loss="loc_weighted", loc_loss_weight=3.0)
        pred_actions = torch.zeros((1, 1, 18), dtype=torch.float32)
        velocity = torch.zeros((1, 1, 18), dtype=torch.float32)
        velocity[:, :, 0] = 1.0
        action_mask = torch.ones((1, 1, 18), dtype=torch.float32)
        action_mask[:, :, 0] = 0.0

        loss = head._compute_loss(pred_actions, velocity, action_mask)

        self.assertAlmostEqual(loss.item(), 0.0)

    def test_validate_loss_config_rejects_non_positive_location_weight(self):
        head = self._make_head(loss="original", loc_loss_weight=0.0)

        with self.assertRaisesRegex(ValueError, "loc_loss_weight must be positive"):
            head._validate_loss_config()

    def test_validate_loss_config_rejects_wrong_action_dim_for_loc_weighted(self):
        head = self._make_head(loss="loc_weighted", loc_loss_weight=3.0, action_dim=17)

        with self.assertRaisesRegex(ValueError, "action_dim == 18"):
            head._validate_loss_config()


if __name__ == "__main__":
    unittest.main()
