#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example local generation refine adapter")
    parser.add_argument("--identity-pack", required=True, dest="identity_pack")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from generation_contract import (  # imported late so script works with PYTHONPATH=/worker/src
        CONTRACT_VERSION,
        AdapterReport,
        load_identity_pack,
        save_adapter_report,
    )

    identity_pack = load_identity_pack(Path(args.identity_pack))
    if not Path(args.input).exists():
        raise SystemExit(f"input video does not exist: {args.input}")

    shutil.copyfile(args.input, args.output)

    if args.report:
        save_adapter_report(
            Path(args.report),
            AdapterReport(
                version=CONTRACT_VERSION,
                stage="identity_refine",
                engine="example-local-refine",
                model="passthrough-demo",
                metrics={"image_count": len(identity_pack.images)},
                warnings=["Example refiner is a passthrough copy. Replace with identity refinement pipeline."],
            ),
        )


if __name__ == "__main__":
    main()
