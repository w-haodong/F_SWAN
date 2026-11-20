"""Run F-SWAN on a single radiograph and report Cobb angles and corners."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="PNG or JPEG radiograph")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Compatible F-SWAN checkpoint")
    parser.add_argument("--dino-checkpoint", type=Path, required=True, help="DINOv3 ViT-B pretrained weights")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument("--output", type=Path, help="Write JSON to this file instead of standard output")
    args = parser.parse_args(argv)

    for flag, path in (
        ("--image", args.image),
        ("--checkpoint", args.checkpoint),
        ("--dino-checkpoint", args.dino_checkpoint),
    ):
        if not path.is_file():
            parser.error(f"{flag}: file not found: {path}")
        if args.output is not None and args.output.resolve() == path.resolve():
            parser.error(f"--output cannot overwrite {flag}")

    try:
        from fswan_inference import predict

        result = predict(args.image, args.checkpoint, args.dino_checkpoint, args.device)
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(output, end="")
        else:
            args.output.write_text(output, encoding="utf-8")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"F-SWAN inference failed: {exc}\n")


if __name__ == "__main__":
    main()
