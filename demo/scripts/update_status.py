"""Update the demo ledger from a completed 3BPA pilot artifact.

The script intentionally does not turn a result into a claim of success: it
only reports that the pilot completed and links the metrics available in its
versioned artifact.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="public/status.json")
    parser.add_argument("--run-uri", required=True)
    parser.add_argument("--state", default="completed")
    args = parser.parse_args()
    payload = {
        "nav": "3BPA L4 pilot {}".format(args.state),
        "state": args.state,
        "detail": "The 300 K ensemble and held-out 300/600/1200 K evaluations are archived at {}. Results require seed and baseline comparison before a performance claim.".format(args.run_uri),
        "claim": "results archived",
    }
    output = Path(args.status)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
