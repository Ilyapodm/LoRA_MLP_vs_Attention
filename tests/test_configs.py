"""Protocol arithmetic and target selection; runs without ML dependencies."""
import unittest
from configs import CONFIGS, LR_GRID, FINAL_SEEDS, targets


class ConfigTests(unittest.TestCase):
    def test_parameter_budget_and_shapes(self):
        for name, config in CONFIGS.items():
            with self.subTest(config=name):
                chosen = targets(config)
                actual = sum(rank * (n_in + n_out) for rank, n_in, n_out in chosen.values())
                self.assertEqual(actual, config.lora_parameters)
                self.assertEqual(config.trainable_parameters, 77669 if name == "LP" else 1552229)
                self.assertEqual(len(chosen), {"LP": 0, "A-iso": 48, "M-iso": 24, "B-iso": 72}[name])
                for i in range(12):
                    self.assertEqual(sum(f".layer.{i}." in k for k in chosen), len(chosen) // 12)
        both = targets(CONFIGS["B-iso"])
        self.assertEqual(sum(r * (a + b) for n, (r, a, b) in both.items() if ".attention." in n), 737280)
        self.assertEqual(sum(r * (a + b) for n, (r, a, b) in both.items() if ".attention." not in n), 737280)

    def test_queue_budget(self):
        self.assertEqual(LR_GRID, (1e-4, 1e-3, 1e-2))
        self.assertEqual(len(CONFIGS) * len(LR_GRID) + len(CONFIGS) * len(FINAL_SEEDS), 24)


if __name__ == "__main__":
    unittest.main()
