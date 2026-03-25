import csv
import shutil
import unittest
import uuid
from pathlib import Path

from scripts.run_eval_replacements_parallel import (
    SCALE_CONFIGS,
    build_tasks,
    export_eval_curve_from_log,
)


class RunEvalReplacementsParallelTests(unittest.TestCase):
    def temp_root(self) -> Path:
        root = Path(__file__).resolve().parents[1] / "results" / "_tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: root.exists() and shutil.rmtree(root, ignore_errors=True))
        return root

    def test_build_tasks_covers_all_solver_scale_seed_alpha_combinations(self):
        exe = Path("build/Release/CED_Schedule.exe")
        root = Path(".")

        tasks = build_tasks(root=root, exe=exe, max_evals=400000, log_every=1000)

        self.assertEqual(len(tasks), 180)
        groups = {(t["group"], t["scale"], t["seed"], t["alpha"]) for t in tasks}
        self.assertIn(("lpsr", "T100", 1, 0.2), groups)
        self.assertIn(("lpsr", "T500", 10, 0.8), groups)
        self.assertIn(("nlpsr", "T200", 4, 0.5), groups)

    def test_export_eval_curve_from_log_writes_expected_named_csv(self):
        root = self.temp_root()
        log_path = root / "LPSR_T100_a0.5_s1.log"
        log_path.write_text(
            "\n".join(
                [
                    "Eval 1000: best_fit = 0.91 pop = 40 archive = 20",
                    "Eval 2000: best_fit = 0.83 pop = 40 archive = 20",
                    "Eval 3000: best_fit = 0.80 pop = 40 archive = 20",
                    "The best solution = 0.79",
                ]
            ),
            encoding="utf-8",
        )
        out_path = root / "results" / "eval" / "lpsr" / "alpha0.5" / "T100" / "LPSR_T100_s1_eval.csv"

        export_eval_curve_from_log(log_path, out_path)

        with out_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(
            rows,
            [
                {"eval_count": "1000", "best_fitness": "0.91"},
                {"eval_count": "2000", "best_fitness": "0.83"},
                {"eval_count": "3000", "best_fitness": "0.8"},
            ],
        )

    def test_scale_configs_still_match_expected_problem_names(self):
        self.assertEqual([cfg["name"] for cfg in SCALE_CONFIGS], ["T100", "T200", "T500"])


if __name__ == "__main__":
    unittest.main()
