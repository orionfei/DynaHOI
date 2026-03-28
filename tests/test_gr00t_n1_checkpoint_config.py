import unittest
from pathlib import Path

from gr00t.model.gr00t_n1 import GR00T_N1_5


CHECKPOINT_PATH = Path("/data1/yfl_data/DynaHOI/gr00t/checkpoints/GR00T/temp/checkpoint-100")


class Gr00tCheckpointConfigReconciliationTest(unittest.TestCase):
    def test_reconciles_stale_18dim_checkpoint_config(self):
        config = GR00T_N1_5.config_class.from_pretrained(str(CHECKPOINT_PATH))

        self.assertEqual(config.action_dim, 32)
        self.assertEqual(config.action_head_cfg["action_dim"], 32)
        self.assertEqual(config.action_head_cfg["max_action_dim"], 32)
        self.assertEqual(config.action_head_cfg["max_state_dim"], 64)

        reconciled_config = GR00T_N1_5._reconcile_pretrained_config(CHECKPOINT_PATH, config)

        self.assertEqual(reconciled_config.action_dim, 18)
        self.assertEqual(reconciled_config.action_head_cfg["action_dim"], 18)
        self.assertEqual(reconciled_config.action_head_cfg["max_action_dim"], 18)
        self.assertEqual(reconciled_config.action_head_cfg["max_state_dim"], 18)


if __name__ == "__main__":
    unittest.main()
