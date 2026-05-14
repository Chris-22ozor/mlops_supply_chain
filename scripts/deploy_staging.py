"""Create a governed staging deployment bundle for the trained model."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "training_metrics.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yaml"
STAGING_DIR = PROJECT_ROOT / "artifacts" / "staging"
MANIFEST_PATH = STAGING_DIR / "deployment_manifest.json"


def main() -> None:
    _require_file(MODEL_PATH, "trained model")
    _require_file(METRICS_PATH, "training metrics")
    _require_file(CONFIG_PATH, "settings config")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    promotion_passed = bool(
        metrics.get("promotion_check", {}).get("beats_baseline_mae_by_10pct", False)
    )

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    deployed_model = STAGING_DIR / "model.joblib"
    deployed_metrics = STAGING_DIR / "training_metrics.json"
    deployed_config = STAGING_DIR / "settings.yaml"

    shutil.copy2(MODEL_PATH, deployed_model)
    shutil.copy2(METRICS_PATH, deployed_metrics)
    shutil.copy2(CONFIG_PATH, deployed_config)

    manifest = {
        "deployment_stage": "staging",
        "deployment_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_artifact": str(deployed_model.relative_to(PROJECT_ROOT)),
        "metrics_artifact": str(deployed_metrics.relative_to(PROJECT_ROOT)),
        "config_artifact": str(deployed_config.relative_to(PROJECT_ROOT)),
        "production_promotion_allowed": promotion_passed,
        "production_block_reason": None
        if promotion_passed
        else "Candidate model did not beat the baseline MAE by the required 10%.",
        "baseline_metrics": metrics.get("baseline", {}),
        "candidate_metrics": metrics.get("candidate", {}),
        "promotion_check": metrics.get("promotion_check", {}),
        "allowed_use": [
            "monthly batch scoring",
            "human review workflow testing",
            "shadow evaluation",
            "feedback collection",
        ],
        "disallowed_use": [
            "automatic procurement action",
            "automatic inventory changes",
            "production promotion without human approval",
        ],
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Staging deployment bundle written to {STAGING_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Production promotion allowed: {promotion_passed}")
    if not promotion_passed:
        print(manifest["production_block_reason"])


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


if __name__ == "__main__":
    main()
