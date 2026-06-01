from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
MODULES = [
    "validation.correlations",
    "validation.regression",
    "validation.random_forest",
    "validation.feature_importance",
    "validation.hypothesis_testing",
    "validation.statistical_tests",
]
EXPECTED_OUTPUTS = [
    TABLES / "correlations.csv",
    TABLES / "regression_model_comparison.csv",
    TABLES / "random_forest_model_comparison.csv",
    TABLES / "feature_importance.csv",
    TABLES / "hypothesis_tests.csv",
    TABLES / "statistical_tests.csv",
]


def run_once() -> dict[str, str]:
    for module in MODULES:
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in EXPECTED_OUTPUTS}


def main() -> None:
    first = run_once()
    second = run_once()
    stable = first == second
    report = TABLES / "reproducibility_check.csv"
    report.write_text(
        "file,first_sha256,second_sha256,stable\n"
        + "\n".join(f"{name},{first[name]},{second[name]},{first[name] == second[name]}" for name in first)
        + "\n",
        encoding="utf-8",
    )
    print(f"Validation reproducible: {stable}")
    print(f"Wrote {report}")
    if not stable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
