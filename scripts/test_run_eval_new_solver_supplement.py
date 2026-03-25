import json
import shutil
import unittest
import uuid
from pathlib import Path

from scripts.run_eval_new_solver_supplement import (
    DEFAULT_NEW_SOLVER_ORDER,
    SCALE_CONFIGS,
    auto_detect_new_solvers,
    build_degradation_tasks,
    build_main_tasks,
    build_summary_record,
    build_trace_rows,
    make_alpha_result_from_main,
)


class RunEvalNewSolverSupplementTests(unittest.TestCase):
    def temp_root(self) -> Path:
        root = Path(__file__).resolve().parents[1] / "results" / "_tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: root.exists() and shutil.rmtree(root, ignore_errors=True))
        return root

    def test_auto_detect_new_solvers_prefers_three_registered_candidates(self):
        detected = auto_detect_new_solvers(Path(__file__).resolve().parents[1])
        self.assertEqual(detected, DEFAULT_NEW_SOLVER_ORDER)

    def test_build_main_tasks_creates_three_solver_by_three_scale_grid(self):
        tasks = build_main_tasks(
            root=Path("."),
            exe=Path("build/Release/CED_Schedule.exe"),
            solvers=DEFAULT_NEW_SOLVER_ORDER,
            scales=[cfg for cfg in SCALE_CONFIGS if cfg["name"] in {"T100", "T200", "T500"}],
            seed=1,
            eval_budget=400000,
            log_every=2000,
        )
        self.assertEqual(len(tasks), 9)
        tags = {(task["solver_key"], task["scale"], task["alpha"]) for task in tasks}
        self.assertIn(("rde", "T100", 0.5), tags)
        self.assertIn(("l_srtde", "T200", 0.5), tags)
        self.assertIn(("nl_shade_lbc", "T500", 0.5), tags)

    def test_build_degradation_tasks_skips_nominal_and_covers_24_conditions(self):
        tasks = build_degradation_tasks(
            root=Path("."),
            exe=Path("build/Release/CED_Schedule.exe"),
            solvers=DEFAULT_NEW_SOLVER_ORDER,
            scales=[cfg for cfg in SCALE_CONFIGS if cfg["name"] in {"T200", "T500"}],
            seed=1,
            eval_budget=400000,
            log_every=2000,
        )
        self.assertEqual(len(tasks), 3 * 2 * 12)
        families = {(task["solver_key"], task["scale"], task["scenario_family"], task["severity"]) for task in tasks}
        self.assertIn(("rde", "T200", "cloud_reduction", "r10"), families)
        self.assertIn(("l_srtde", "T500", "edge_reduction", "r30"), families)
        self.assertIn(("nl_shade_lbc", "T500", "communication_inflation", "p60"), families)

    def test_build_trace_rows_preserves_eval_and_elapsed_columns(self):
        root = self.temp_root()
        trace_path = root / "trace.csv"
        trace_path.write_text(
            "\n".join(
                [
                    "eval_count,time_seconds,generation,best_fitness,best_f1,best_f2",
                    "0,0.0,0,1.0,2.0,3.0",
                    "2000,0.5,10,0.8,1.7,2.8",
                ]
            ),
            encoding="utf-8",
        )

        rows = build_trace_rows(trace_path)

        self.assertEqual(
            rows,
            [
                {"eval_count": 0, "best_fitness": 1.0, "elapsed_seconds": 0.0},
                {"eval_count": 2000, "best_fitness": 0.8, "elapsed_seconds": 0.5},
            ],
        )

    def test_make_alpha_result_from_main_marks_reuse_without_rerun(self):
        root = self.temp_root()
        main_path = root / "main.json"
        main_payload = {
            "solver": "RDE",
            "solver_key": "rde",
            "scale": "T100",
            "alpha": 0.5,
            "seed": 1,
            "final_fitness": 0.123,
            "wall_clock_seconds": 9.5,
            "trace_path": "results/eval/traces/rde_T100_main_alpha0.5_seed1.csv",
        }
        main_path.write_text(json.dumps(main_payload), encoding="utf-8")

        reused = make_alpha_result_from_main(main_path, root / "alpha.json")

        self.assertEqual(reused["reused_from"], str(main_path))
        self.assertEqual(reused["solver_key"], "rde")
        self.assertEqual(reused["alpha"], 0.5)

    def test_build_summary_record_includes_nominal_and_prr_for_degradation(self):
        record = build_summary_record(
            payload={
                "solver": "RDE",
                "solver_key": "rde",
                "scale": "T200",
                "seed": 1,
                "alpha": 0.5,
                "scenario_family": "cloud_reduction",
                "severity": "r10",
                "scenario_label": "cloud_reduction_r10",
                "final_fitness": 0.4,
                "wall_clock_seconds": 12.0,
                "nominal_fitness": 0.2,
                "nominal_wall_clock_seconds": 7.0,
                "prr": 2.0,
                "trace_path": "trace.csv",
                "result_path": "result.json",
            }
        )

        self.assertEqual(record["nominal_fitness"], 0.2)
        self.assertEqual(record["degraded_fitness"], 0.4)
        self.assertEqual(record["prr"], 2.0)


if __name__ == "__main__":
    unittest.main()
