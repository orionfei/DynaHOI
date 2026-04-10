import unittest

import tyro

from scripts.finetune_policy import TrainArgsConfig


class FinetuneLossConfigTest(unittest.TestCase):
    def test_loss_defaults_to_original(self):
        config = tyro.cli(TrainArgsConfig, args=["--dataset-path", "dummy"])

        self.assertEqual(config.loss, "original")
        self.assertEqual(config.loc_loss_weight, 3.0)

    def test_loss_cli_accepts_loc_weighted(self):
        config = tyro.cli(
            TrainArgsConfig,
            args=[
                "--dataset-path",
                "dummy",
                "--loss",
                "loc_weighted",
                "--loc-loss-weight",
                "3.0",
            ],
        )

        self.assertEqual(config.loss, "loc_weighted")
        self.assertEqual(config.loc_loss_weight, 3.0)


if __name__ == "__main__":
    unittest.main()
