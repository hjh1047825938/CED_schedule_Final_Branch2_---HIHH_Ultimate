import csv
import unittest
from pathlib import Path
import uuid
import shutil

from scripts.convert_rerun_logs_to_eval import (
    build_eval_curve,
    convert_cchihh_operator_tables,
    convert_cchihh_weight_tables,
    convert_tree,
    parse_cchihh_csv_metadata,
    parse_input_metadata,
    parse_series_from_log,
)


class ConvertRerunLogsToEvalTests(unittest.TestCase):
    def temp_root(self) -> Path:
        root = Path(__file__).resolve().parents[1] / "results" / "_tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: root.exists() and shutil.rmtree(root, ignore_errors=True))
        return root

    def test_parse_run_metadata_for_alpha_run(self):
        path = Path("results/rerun_alpha/cchihh_full_T200_a0.2_s7.log")

        meta = parse_input_metadata(path)

        self.assertEqual(meta.group, "rerun_alpha")
        self.assertEqual(meta.variant, "cchihh_full")
        self.assertEqual(meta.scale, "T200")
        self.assertEqual(meta.alpha, 0.2)
        self.assertEqual(meta.seed, 7)

    def test_parse_input_metadata_for_dsac_baseline_run(self):
        path = Path("outputs/results/dsac_de_multiscale/T500/DSAC_DE_seed3.txt")

        meta = parse_input_metadata(path)

        self.assertEqual(meta.group, "dsac_de_multiscale")
        self.assertEqual(meta.variant, "DSAC_DE")
        self.assertEqual(meta.scale, "T500")
        self.assertEqual(meta.alpha, 0.5)
        self.assertEqual(meta.seed, 3)

    def test_parse_input_metadata_for_dsac_alpha_run(self):
        path = Path("outputs/results/alpha_sensitivity_cga_imoma_dsac/runs/alpha_0.2/T100/DSAC_DE_seed4.txt")

        meta = parse_input_metadata(path)

        self.assertEqual(meta.group, "dsac_alpha")
        self.assertEqual(meta.variant, "DSAC_DE")
        self.assertEqual(meta.scale, "T100")
        self.assertEqual(meta.alpha, 0.2)
        self.assertEqual(meta.seed, 4)

    def test_parse_input_metadata_for_imoma_baseline_run(self):
        path = Path("outputs/results/cchihh_ablation_suite/baseline/T200/IMOMA_seed6.txt")

        meta = parse_input_metadata(path)

        self.assertEqual(meta.group, "imoma_baseline")
        self.assertEqual(meta.variant, "IMOMA")
        self.assertEqual(meta.scale, "T200")
        self.assertEqual(meta.alpha, 0.5)
        self.assertEqual(meta.seed, 6)

    def test_parse_input_metadata_for_imoma_alpha_run(self):
        path = Path("outputs/results/alpha_sensitivity_cga_imoma_dsac/runs/alpha_0.8/T500/IMOMA_seed9.txt")

        meta = parse_input_metadata(path)

        self.assertEqual(meta.group, "imoma_alpha")
        self.assertEqual(meta.variant, "IMOMA")
        self.assertEqual(meta.scale, "T500")
        self.assertEqual(meta.alpha, 0.8)
        self.assertEqual(meta.seed, 9)

    def test_parse_input_metadata_for_cga_baseline_run(self):
        path = Path("outputs/results/cchihh_ablation_suite/baseline/T100/CGA_seed2.txt")

        meta = parse_input_metadata(path)

        self.assertEqual(meta.group, "cga_baseline")
        self.assertEqual(meta.variant, "CGA")
        self.assertEqual(meta.scale, "T100")
        self.assertEqual(meta.alpha, 0.5)
        self.assertEqual(meta.seed, 2)

    def test_parse_input_metadata_for_cga_alpha_run(self):
        path = Path("outputs/results/alpha_sensitivity_cga_imoma_dsac/runs/alpha_0.2/T200/CGA_seed8.txt")

        meta = parse_input_metadata(path)

        self.assertEqual(meta.group, "cga_alpha")
        self.assertEqual(meta.variant, "CGA")
        self.assertEqual(meta.scale, "T200")
        self.assertEqual(meta.alpha, 0.2)
        self.assertEqual(meta.seed, 8)

    def test_parse_input_metadata_for_ppo_baseline_run(self):
        path = Path("outputs/results/cchihh_ablation_suite/baseline/T500/PPO_seed3.txt")

        meta = parse_input_metadata(path)

        self.assertEqual(meta.group, "ppo_baseline")
        self.assertEqual(meta.variant, "PPO")
        self.assertEqual(meta.scale, "T500")
        self.assertEqual(meta.alpha, 0.5)
        self.assertEqual(meta.seed, 3)

    def test_parse_series_from_log_reads_generation_points(self):
        root = self.temp_root()
        log_path = root / "sample.log"
        log_path.write_text(
            "\n".join(
                [
                    "Gen 50: best_fit = 0.80",
                    "Gen 100: best_fit = 0.70",
                    "The best solution = 0.65",
                ]
            ),
            encoding="utf-8",
        )

        series = parse_series_from_log(log_path)

        self.assertEqual(series, {50: 0.8, 100: 0.7})

    def test_parse_cchihh_csv_metadata_for_operator_table(self):
        path = Path("results/rerun_full/cchihh_full_T200_s7_ops.csv")

        meta = parse_cchihh_csv_metadata(path)

        self.assertEqual(meta.group, "rerun_full")
        self.assertEqual(meta.variant, "cchihh_full")
        self.assertEqual(meta.scale, "T200")
        self.assertEqual(meta.alpha, 0.5)
        self.assertEqual(meta.seed, 7)

    def test_build_eval_curve_resamples_at_fixed_generation_interval(self):
        curve = build_eval_curve(
            series={50: 0.9, 100: 0.8, 150: 0.85, 200: 0.7},
            gen_interval=100,
            max_eval_budget=30000,
            init_evals=40,
            evals_per_gen=120,
        )

        self.assertEqual(curve, [(12040, 0.8), (24040, 0.7)])

    def test_convert_tree_writes_categorized_csv_outputs(self):
        root = self.temp_root()
        input_root = root / "results"
        (input_root / "rerun_full").mkdir(parents=True, exist_ok=True)
        log_path = input_root / "rerun_full" / "cchihh_full_T100_s1.log"
        log_path.write_text(
            "\n".join(
                [
                    "Gen 50: best_fit = 1.0",
                    "Gen 100: best_fit = 0.9",
                    "Gen 150: best_fit = 0.85",
                    "Time = 12.3 s",
                ]
            ),
            encoding="utf-8",
        )
        output_root = input_root / "eval"

        written = convert_tree(
            results_root=input_root,
            output_root=output_root,
            gen_interval=50,
            max_eval_budget=20000,
            init_evals=40,
            evals_per_gen=120,
        )

        self.assertEqual(written, 1)
        csv_path = output_root / "rerun_full" / "alpha0.5" / "T100" / "cchihh_full_T100_s1_eval.csv"
        self.assertTrue(csv_path.exists())

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(
            rows,
            [
                {"eval_count": "6040", "best_fitness": "1"},
                {"eval_count": "12040", "best_fitness": "0.9"},
                {"eval_count": "18040", "best_fitness": "0.85"},
            ],
        )

        index_path = output_root / "index.csv"
        self.assertTrue(index_path.exists())

    def test_convert_tree_writes_dsac_outputs_with_dsac_eval_rule(self):
        root = self.temp_root()
        dsac_root = root / "outputs" / "results" / "dsac_de_multiscale" / "T100"
        dsac_root.mkdir(parents=True, exist_ok=True)
        log_path = dsac_root / "DSAC_DE_seed1.txt"
        log_path.write_text(
            "\n".join(
                [
                    "Gen 50: best_fit = 1.0",
                    "Gen 100: best_fit = 0.8",
                    "Gen 150: best_fit = 0.75",
                ]
            ),
            encoding="utf-8",
        )

        output_root = root / "results" / "eval"
        written = convert_tree(
            results_root=root / "results",
            output_root=output_root,
            gen_interval=50,
            max_eval_budget=10000,
            init_evals=40,
            evals_per_gen=120,
            extra_input_roots=[root / "outputs" / "results"],
        )

        self.assertEqual(written, 1)
        csv_path = output_root / "dsac_de_multiscale" / "alpha0.5" / "T100" / "DSAC_DE_T100_s1_eval.csv"
        self.assertTrue(csv_path.exists())

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(
            rows,
            [
                {"eval_count": "2040", "best_fitness": "1"},
                {"eval_count": "4040", "best_fitness": "0.8"},
                {"eval_count": "6040", "best_fitness": "0.75"},
            ],
        )

    def test_convert_tree_writes_imoma_outputs_with_estimated_eval_rule(self):
        root = self.temp_root()
        imoma_root = root / "outputs" / "results" / "cchihh_ablation_suite" / "baseline" / "T100"
        imoma_root.mkdir(parents=True, exist_ok=True)
        log_path = imoma_root / "IMOMA_seed1.txt"
        log_path.write_text(
            "\n".join(
                [
                    "Gen 50: best_fit = 1.0",
                    "Gen 100: best_fit = 0.95",
                    "Gen 150: best_fit = 0.9",
                ]
            ),
            encoding="utf-8",
        )

        output_root = root / "results" / "eval"
        written = convert_tree(
            results_root=root / "results",
            output_root=output_root,
            gen_interval=50,
            max_eval_budget=100000,
            init_evals=40,
            evals_per_gen=120,
            extra_input_roots=[root / "outputs" / "results"],
        )

        self.assertEqual(written, 1)
        csv_path = output_root / "imoma_baseline" / "alpha0.5" / "T100" / "IMOMA_T100_s1_eval.csv"
        self.assertTrue(csv_path.exists())

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(
            rows,
            [
                {"eval_count": "10080", "best_fitness": "1"},
                {"eval_count": "20080", "best_fitness": "0.95"},
                {"eval_count": "30080", "best_fitness": "0.9"},
            ],
        )

    def test_convert_tree_writes_cga_outputs_with_cga_eval_rule(self):
        root = self.temp_root()
        cga_root = root / "outputs" / "results" / "cchihh_ablation_suite" / "baseline" / "T100"
        cga_root.mkdir(parents=True, exist_ok=True)
        log_path = cga_root / "CGA_seed1.txt"
        log_path.write_text(
            "\n".join(
                [
                    "Gen 50: best_fit = 1.0",
                    "Gen 100: best_fit = 0.92",
                    "Gen 150: best_fit = 0.88",
                ]
            ),
            encoding="utf-8",
        )

        output_root = root / "results" / "eval"
        written = convert_tree(
            results_root=root / "results",
            output_root=output_root,
            gen_interval=50,
            max_eval_budget=100000,
            init_evals=40,
            evals_per_gen=120,
            extra_input_roots=[root / "outputs" / "results"],
        )

        self.assertEqual(written, 1)
        csv_path = output_root / "cga_baseline" / "alpha0.5" / "T100" / "CGA_T100_s1_eval.csv"
        self.assertTrue(csv_path.exists())

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(
            rows,
            [
                {"eval_count": "2040", "best_fitness": "1"},
                {"eval_count": "4040", "best_fitness": "0.92"},
                {"eval_count": "6040", "best_fitness": "0.88"},
            ],
        )

    def test_convert_tree_writes_ppo_outputs_with_episode_eval_rule(self):
        root = self.temp_root()
        ppo_root = root / "outputs" / "results" / "cchihh_ablation_suite" / "baseline" / "T100"
        ppo_root.mkdir(parents=True, exist_ok=True)
        log_path = ppo_root / "PPO_seed1.txt"
        log_path.write_text(
            "\n".join(
                [
                    "Gen 50: best_fit = 1.0",
                    "Gen 100: best_fit = 0.8",
                    "Gen 150: best_fit = 0.75",
                ]
            ),
            encoding="utf-8",
        )

        output_root = root / "results" / "eval"
        written = convert_tree(
            results_root=root / "results",
            output_root=output_root,
            gen_interval=50,
            max_eval_budget=100000,
            init_evals=40,
            evals_per_gen=120,
            extra_input_roots=[root / "outputs" / "results"],
        )

        self.assertEqual(written, 1)
        csv_path = output_root / "ppo_baseline" / "alpha0.5" / "T100" / "PPO_T100_s1_eval.csv"
        self.assertTrue(csv_path.exists())

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(
            rows,
            [
                {"eval_count": "2000", "best_fitness": "1"},
                {"eval_count": "4000", "best_fitness": "0.8"},
                {"eval_count": "6000", "best_fitness": "0.75"},
            ],
        )

    def test_convert_cchihh_operator_tables_writes_eval_keyed_csv(self):
        root = self.temp_root()
        rerun_root = root / "results" / "rerun_full"
        rerun_root.mkdir(parents=True, exist_ok=True)
        ops_path = rerun_root / "cchihh_full_T100_s1_ops.csv"
        ops_path.write_text(
            "\n".join(
                [
                    "gen,offload_GA,offload_DE",
                    "50,0.3,0.7",
                    "100,0.4,0.6",
                    "3400,0.5,0.5",
                ]
            ),
            encoding="utf-8",
        )

        output_root = root / "results" / "eval"
        written = convert_cchihh_operator_tables(
            results_root=root / "results",
            output_root=output_root,
            max_eval_budget=400000,
            init_evals=40,
            evals_per_gen=120,
        )

        self.assertEqual(written, 1)
        csv_path = output_root / "rerun_full" / "alpha0.5" / "T100" / "cchihh_full_T100_s1_ops_eval.csv"
        self.assertTrue(csv_path.exists())

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(
            rows,
            [
                {"eval_count": "6040", "offload_GA": "0.3", "offload_DE": "0.7"},
                {"eval_count": "12040", "offload_GA": "0.4", "offload_DE": "0.6"},
            ],
        )

    def test_convert_cchihh_weight_tables_writes_eval_keyed_csv(self):
        root = self.temp_root()
        rerun_root = root / "results" / "rerun_full"
        rerun_root.mkdir(parents=True, exist_ok=True)
        weight_path = rerun_root / "cchihh_full_T100_s1_w_off.csv"
        weight_path.write_text(
            "\n".join(
                [
                    "gen,op_id,w0,norm",
                    "50,0,0.1,0.1",
                    "100,1,0.2,0.2",
                    "3400,2,0.3,0.3",
                ]
            ),
            encoding="utf-8",
        )

        output_root = root / "results" / "eval"
        written = convert_cchihh_weight_tables(
            results_root=root / "results",
            output_root=output_root,
            max_eval_budget=400000,
            init_evals=40,
            evals_per_gen=120,
        )

        self.assertEqual(written, 1)
        csv_path = output_root / "rerun_full" / "alpha0.5" / "T100" / "cchihh_full_T100_s1_w_off_eval.csv"
        self.assertTrue(csv_path.exists())

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(
            rows,
            [
                {"eval_count": "6040", "op_id": "0", "w0": "0.1", "norm": "0.1"},
                {"eval_count": "12040", "op_id": "1", "w0": "0.2", "norm": "0.2"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
