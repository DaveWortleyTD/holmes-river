"""
Read the river gauge level from an image using Claude Haiku vision.

Usage:
    python read_gauge.py latest.jpg
    python read_gauge.py https://example.com/photo.jpg
    python read_gauge.py --spacing 20 latest.jpg
"""

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from src.gauge import read_gauge, MODEL


def main():
    parser = argparse.ArgumentParser(description="Read river gauge level from an image")
    parser.add_argument("image", help="Local image path or URL")
    parser.add_argument("--reference", default=None,
                        help="Reference photo of the full gauge staff (overrides GAUGE_REFERENCE_IMAGE in .env)")
    parser.add_argument("--local-model", action="store_true",
                        help="Use the locally-trained model instead of Claude (run train_model.py first)")
    args = parser.parse_args()

    if args.local_model:
        from src.local_model import predict
        print(f"Image : {args.image}")
        print("Using local trained model\n")
        try:
            level = predict(args.image)
            print(f"Level : {level}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    print(f"Image     : {args.image}")
    print(f"Reference : {args.reference or '(from .env or none)'}")
    print(f"Model     : {MODEL}")
    print("Sending to Claude...\n")

    try:
        reading = read_gauge(args.image, reference_image=args.reference)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Level  : {reading.level}")
    print(f"Confidence : {reading.confidence}")
    print(f"Notes      : {reading.notes}")
    print(f"\nRaw response: {reading.raw_json}")


if __name__ == "__main__":
    main()
