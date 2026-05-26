"""Physical constraint validation for generated designs."""

import numpy as np

from ..utils.physics import validate_spectrum, PARAM_BOUNDS, THICKNESS_INDEX


def validate_design(params: np.ndarray, spectrum: np.ndarray,
                    target_thickness_mm: float) -> dict:
    """Comprehensive design validation.

    Returns dict with keys: valid, issues, warnings, details.
    """
    issues = []
    warnings = []
    details = {}

    # 1. Check parameter bounds
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        if params[i] < lo - 0.01 or params[i] > hi + 0.01:
            issues.append(f"Param[{i}] = {params[i]:.2f} outside bounds [{lo}, {hi}]")

    # 2. Check thickness constraint
    actual_thickness = params[THICKNESS_INDEX]
    thickness_error = abs(actual_thickness - target_thickness_mm)
    details["thickness_error_mm"] = float(thickness_error)
    if thickness_error > 0.1:
        issues.append(
            f"Thickness mismatch: expected {target_thickness_mm}, got {actual_thickness:.2f}"
        )

    # 3. Check spectrum physics
    spec_check = validate_spectrum(spectrum)
    issues.extend(spec_check["issues"])
    details["roughness"] = spec_check["roughness"]
    details["low_f_absorption"] = spec_check["low_f_absorption"]

    # 4. Check for degenerate designs (all params at bounds)
    at_lower = np.sum(params <= (PARAM_BOUNDS[:, 0] + 0.01))
    at_upper = np.sum(params >= (PARAM_BOUNDS[:, 1] - 0.01))
    if at_lower > 20:
        warnings.append(f"Design saturated at lower bounds ({int(at_lower)} params)")
    if at_upper > 20:
        warnings.append(f"Design saturated at upper bounds ({int(at_upper)} params)")
    details["params_at_lower"] = int(at_lower)
    details["params_at_upper"] = int(at_upper)

    valid = len(issues) == 0

    return {
        "valid": valid,
        "issues": issues,
        "warnings": warnings,
        "details": details,
    }
