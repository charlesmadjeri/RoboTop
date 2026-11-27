import argparse
import json
from pathlib import Path

from perception_bench import report, report_motion


def merge(runs: list[dict]) -> dict:
    merged = {"frames": _frames(runs), "methods": {}, "motion": {}}
    for run in runs:
        duplicated = set(run["methods"]) & set(merged["methods"])
        if duplicated:
            raise ValueError(f"method in two runs: {', '.join(sorted(duplicated))}")
        merged["methods"].update(run["methods"])
        merged["motion"].update(run["motion"])
    return merged


def _frames(runs: list[dict]) -> dict:
    names = list(runs[0]["frames"])
    if any(list(run["frames"]) != names for run in runs):
        raise ValueError("the runs do not share the same scenarios")
    counts = {name: sorted({run["frames"][name] for run in runs}) for name in names}
    return {name: c[0] if len(c) == 1 else f"{c[0]} to {c[-1]}" for name, c in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    merged = merge([json.loads(path.read_text()) for path in args.results])
    print(report.render(merged))
    print(report_motion.render(merged["motion"]))


if __name__ == "__main__":
    main()
