"""Lightweight temporal-window indexing for persistent graph datasets."""
from __future__ import annotations

from .config import MODEL_DT, OBS_STEPS, PRED_STEPS, WINDOW_STRIDE


def model_gap(scenario: dict, model_dt: float = MODEL_DT) -> int | None:
    source_dt = float(scenario.get("dt", 0.0))
    if source_dt <= 0:
        return None
    ratio = model_dt / source_dt
    gap = int(round(ratio))
    if gap < 1 or abs(gap * source_dt - model_dt) > 1e-6:
        return None
    return gap


def build_sample_records(
    scenario: dict,
    scenario_file,
    obs_steps: int = OBS_STEPS,
    pred_steps: int = PRED_STEPS,
    model_dt: float = MODEL_DT,
    stride: int = WINDOW_STRIDE,
) -> list[dict]:
    """Create window references without forcing all vehicles to span the window.

    CommonRoadTemporalData can contain a varying number of vehicle nodes at each
    timestep. A sample is valid when at least one vehicle present at the latest
    observation has ground truth for every future prediction step.
    """
    vehicles = scenario.get("vehicles", {})
    gap = model_gap(scenario, model_dt)
    if not vehicles or gap is None:
        return []

    all_times = sorted({t for vehicle in vehicles.values() for t in vehicle.get("states", {})})
    if not all_times:
        return []

    total_steps = obs_steps + pred_steps
    span = (total_steps - 1) * gap
    start_stride = max(1, int(stride)) * gap
    records: list[dict] = []

    for start in range(all_times[0], all_times[-1] - span + 1, start_stride):
        times = tuple(start + k * gap for k in range(total_steps))
        observed = times[:obs_steps]
        future = times[obs_steps:]
        latest = observed[-1]
        latest_ids = [vid for vid, vehicle in vehicles.items() if latest in vehicle["states"]]
        valid_targets = [
            vid
            for vid in latest_ids
            if all(t in vehicles[vid]["states"] for t in future)
        ]
        if not valid_targets:
            continue
        records.append({
            "scenario_file": str(scenario_file),
            "times": times,
        })
    return records
