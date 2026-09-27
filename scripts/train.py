from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medisense.ml.training import train_disease
from medisense.specs import SPECS

parser = argparse.ArgumentParser(description="Train MediSense disease-risk models.")
parser.add_argument("--disease", choices=tuple(SPECS), help="Train one disease model.")
parser.add_argument("--all", action="store_true", help="Train all four models.")
args = parser.parse_args()
if not args.all and not args.disease:
    parser.error("Choose --disease or --all")
for disease in (SPECS if args.all else [args.disease]):
    print(f"Training {disease}...")
    print(json.dumps(train_disease(disease)["test"], indent=2))
