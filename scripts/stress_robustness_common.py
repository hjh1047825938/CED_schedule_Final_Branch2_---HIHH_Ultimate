from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_SCENARIO_LEVELS = {
    "cloud_reduction": [0.10, 0.20, 0.30],
    "edge_reduction": [0.10, 0.20, 0.30],
    "device_reduction": [0.10, 0.20, 0.30],
    "communication_inflation": [0.20, 0.40, 0.60],
}


@dataclass(frozen=True)
class ScenarioDefinition:
    family: str
    severity_label: str
    severity_value: float
    cloud_scale: float = 1.0
    edge_scale: float = 1.0
    device_scale: float = 1.0
    comm_scale: float = 1.0

    @property
    def is_nominal(self) -> bool:
        return self.family == "nominal"


def _reduction_label(level: float) -> str:
    return f"r{int(round(level * 100.0))}"


def _inflation_label(level: float) -> str:
    return f"p{int(round(level * 100.0))}"


def make_scenario_definitions(
    scenario_levels: dict[str, list[float]] | None = None,
) -> list[ScenarioDefinition]:
    levels = dict(DEFAULT_SCENARIO_LEVELS if scenario_levels is None else scenario_levels)
    scenarios = [
        ScenarioDefinition(
            family="nominal",
            severity_label="nominal",
            severity_value=0.0,
        )
    ]

    for level in levels.get("cloud_reduction", []):
        scenarios.append(
            ScenarioDefinition(
                family="cloud_reduction",
                severity_label=_reduction_label(level),
                severity_value=level,
                cloud_scale=1.0 - level,
            )
        )
    for level in levels.get("edge_reduction", []):
        scenarios.append(
            ScenarioDefinition(
                family="edge_reduction",
                severity_label=_reduction_label(level),
                severity_value=level,
                edge_scale=1.0 - level,
            )
        )
    for level in levels.get("device_reduction", []):
        scenarios.append(
            ScenarioDefinition(
                family="device_reduction",
                severity_label=_reduction_label(level),
                severity_value=level,
                device_scale=1.0 - level,
            )
        )
    for level in levels.get("communication_inflation", []):
        scenarios.append(
            ScenarioDefinition(
                family="communication_inflation",
                severity_label=_inflation_label(level),
                severity_value=level,
                comm_scale=1.0 + level,
            )
        )
    return scenarios


def non_nominal_scenarios(
    scenario_levels: dict[str, list[float]] | None = None,
) -> Iterable[ScenarioDefinition]:
    return [item for item in make_scenario_definitions(scenario_levels) if not item.is_nominal]
