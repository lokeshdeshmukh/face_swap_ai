#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic generation refine stage")
    parser.add_argument("--identity-pack", required=True, dest="identity_pack")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from generation_contract import (
        CONTRACT_VERSION,
        AdapterReport,
        ensure_video_output,
        load_identity_pack,
        save_adapter_report,
    )

    identity_pack = load_identity_pack(Path(args.identity_pack))
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"input video does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
    ensure_video_output(output_path)

    warnings: list[str] = []
    if len(identity_pack.images) > 1:
        warnings.append("Identity refinement is currently a passthrough stage; multi-image consistency is not yet applied.")

    if args.report:
        save_adapter_report(
            Path(args.report),
            AdapterReport(
                version=CONTRACT_VERSION,
                stage="identity_refine",
                engine="basic_passthrough",
                model="copy",
                metrics={"image_count": len(identity_pack.images)},
                warnings=warnings,
            ),
        )


if __name__ == "__main__":
    main()
