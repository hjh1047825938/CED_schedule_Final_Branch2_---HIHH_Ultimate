import math
import unittest
from pathlib import Path

from scripts.analyze_stress_robustness import (
    compute_degradation_percent,
    compute_improvement_percent,
    make_scenario_definitions,
)
from scripts.run_stress_robustness_eval import (
    ALGORITHMS,
    DEFAULT_SCENARIO_LEVELS,
    SCALE_CONFIGS,
    build_tasks,
)


class StressRobustnessTests(unittest.TestCase):
    def test_make_scenario_definitions_includes_nominal_and_default_families(self):
        scenarios = make_scenario_definitions(DEFAULT_SCENARIO_LEVELS)
        names = [(item.family, item.severity_label) for item in scenarios]

        self.assertEqual(names[0], ("nominal", "nominal"))
        self.assertIn(("cloud_reduction", "r10"), names)
        self.assertIn(("cloud_reduction", "r20"), names)
        self.assertIn(("cloud_reduction", "r30"), names)
        self.assertIn(("edge_reduction", "r10"), names)
        self.assertIn(("device_reduction", "r30"), names)
        self.assertIn(("communication_inflation", "p60"), names)

    def test_build_tasks_covers_t200_t500_10_seeds_all_algorithms_all_scenarios(self):
        root = Path(".")
        exe = Path("build/Release/CED_Schedule.exe")
        tasks_all = build_tasks(
            root=root,
            exe=exe,
            scales=[cfg for cfg in SCALE_CONFIGS if cfg["name"] in {"T200", "T500"}],
            algorithms=ALGORITHMS,
            seeds=list(range(1, 11)),
            scenario_levels=DEFAULT_SCENARIO_LEVELS,
            include_shared_bandit=False,
            reuse_nominal=False,
        )
        tasks_stress_only = build_tasks(
            root=root,
            exe=exe,
            scales=[cfg for cfg in SCALE_CONFIGS if cfg["name"] in {"T200", "T500"}],
            algorithms=ALGORITHMS,
            seeds=list(range(1, 11)),
            scenario_levels=DEFAULT_SCENARIO_LEVELS,
            include_shared_bandit=False,
            reuse_nominal=True,
        )

        # 2 scales * 4 algorithms * 10 seeds * (1 nominal + 4*3 stress levels)
        self.assertEqual(len(tasks_all), 2 * 4 * 10 * 13)
        # stress-only mode reuses nominal rows and launches only 12 stress conditions
        self.assertEqual(len(tasks_stress_only), 2 * 4 * 10 * 12)
        tags = {
            (task["instance"], task["algorithm"], task["scenario_family"], task["severity"], task["seed"])
            for task in tasks_all
        }
        self.assertIn(("T200", "CCHIHH", "nominal", "nominal", 1), tags)
        self.assertIn(("T500", "DSAC-DE", "communication_inflation", "p60", 10), tags)
        self.assertIn(("T200", "IMOMA", "device_reduction", "r20", 4), tags)

    def test_degradation_percent_uses_nominal_reference_for_minimization(self):
        degradation = compute_degradation_percent(stress_mean=12.0, nominal_mean=10.0)
        self.assertAlmostEqual(degradation, 20.0)

    def test_improvement_percent_is_positive_when_cchihh_is_better(self):
        improvement = compute_improvement_percent(cchihh_mean=9.0, baseline_mean=12.0)
        self.assertAlmostEqual(improvement, 25.0)

    def test_improvement_percent_is_zero_for_identical_means(self):
        self.assertTrue(math.isclose(compute_improvement_percent(5.0, 5.0), 0.0))


if __name__ == "__main__":
    unittest.main()
