import math
import unittest
from pathlib import Path
import uuid

import numpy as np

from scripts.wallclock_common import aggregate_curves, load_curve_csv


class WallclockCommonTests(unittest.TestCase):
    def temp_root(self) -> Path:
        root = Path(__file__).resolve().parents[1] / "results" / "_tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: root.exists() and __import__("shutil").rmtree(root, ignore_errors=True))
        return root

    def write_csv(self, root: Path, name: str, rows: list[tuple[float, int, float, float, float]]) -> Path:
        path = root / name
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("time_seconds,generation,best_fitness,best_f1,best_f2\n")
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")
        return path

    def test_load_curve_csv_preserves_initial_row(self):
        root = self.temp_root()
        path = self.write_csv(
            root,
            "seed1.csv",
            [
                (0.0, 0, 1.0, 10.0, 20.0),
                (0.5, 1, 0.8, 9.0, 19.0),
            ],
        )

        curve = load_curve_csv(path)

        self.assertEqual(curve.time_seconds[0], 0.0)
        self.assertEqual(curve.generation[0], 0)
        self.assertEqual(curve.best_fitness[0], 1.0)
        self.assertEqual(curve.best_f1[0], 10.0)
        self.assertEqual(curve.best_f2[0], 20.0)

    def test_aggregate_curves_interpolates_and_extrapolates(self):
        root = self.temp_root()
        p1 = self.write_csv(
            root,
            "seed1.csv",
            [
                (0.0, 0, 1.0, 10.0, 20.0),
                (1.0, 1, 0.5, 9.0, 18.0),
            ],
        )
        p2 = self.write_csv(
            root,
            "seed2.csv",
            [
                (0.0, 0, 0.9, 11.0, 21.0),
                (0.5, 1, 0.7, 10.0, 20.0),
            ],
        )

        summary = aggregate_curves([p1, p2], time_budget=2.0, n_grid=5)

        self.assertEqual(len(summary.grid), 5)
        self.assertTrue(np.allclose(summary.grid, np.linspace(0.0, 2.0, 5)))
        self.assertTrue(np.allclose(summary.per_seed[0], [1.0, 0.75, 0.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(summary.per_seed[1], [0.9, 0.7, 0.7, 0.7, 0.7]))
        self.assertTrue(np.allclose(summary.mean, [0.95, 0.725, 0.6, 0.6, 0.6]))

        expected_std = np.std(np.array([[1.0, 0.75, 0.5, 0.5, 0.5], [0.9, 0.7, 0.7, 0.7, 0.7]]), axis=0, ddof=1)
        expected_ci95 = 1.96 * expected_std / math.sqrt(2.0)
        self.assertTrue(np.allclose(summary.ci95, expected_ci95))


if __name__ == "__main__":
    unittest.main()
