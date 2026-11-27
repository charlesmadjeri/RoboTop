from collections.abc import Callable

Formatter = Callable[[float | None], str]


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _decimal(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


SECTIONS = (
    ("Cube recall", "cube", "recall", min, _percent),
    ("Cube p95 error (px)", "cube", "p95_px", max, _decimal),
    ("Cube false positives", "cube", "false_positive", max, _percent),
    ("Box recall", "box", "recall", min, _percent),
    ("Box false positives", "box", "false_positive", max, _percent),
)
LATENCY_HEADING = "Latency (ms per frame)"
FRAMES_HEADING = "Frames per scenario"


def render(results: dict) -> str:
    frames = list(results["frames"])
    methods = list(results["methods"])
    blocks = [
        _section(title, _metric_table(results, frames, methods, obj, key, pick, fmt))
        for title, obj, key, pick, fmt in SECTIONS
    ]
    blocks.append(_section(LATENCY_HEADING, _latency_table(results, methods)))
    blocks.append(_section(FRAMES_HEADING, _frames_table(results)))
    return "\n\n".join(blocks) + "\n"


def _frames_table(results: dict) -> list[str]:
    return _grid(["Scenario", "Frames"], [[name, str(count)] for name, count in results["frames"].items()])


def _section(title: str, table: list[str]) -> str:
    return "\n".join([f"### {title}", "", *table])


def _metric_table(
    results: dict,
    frames: list[str],
    methods: list[str],
    obj: str,
    key: str,
    pick: Callable[[list[float]], float],
    fmt: Formatter,
) -> list[str]:
    values = {
        (frame, method): _metric_value(results, method, frame, obj, key)
        for frame in frames
        for method in methods
    }
    rows = [[frame, *(fmt(values[frame, method]) for method in methods)] for frame in frames]
    worst = []
    for method in methods:
        present = [values[frame, method] for frame in frames if values[frame, method] is not None]
        worst.append(fmt(pick(present)) if present else "n/a")
    rows.append(["worst", *worst])
    return _grid(["Scenario", *methods], rows)


def _latency_table(results: dict, methods: list[str]) -> list[str]:
    resolutions = list(dict.fromkeys(res for method in methods for res in results["methods"][method]["latency_ms"]))
    rows = [
        [res, *(_decimal(results["methods"][method]["latency_ms"].get(res)) for method in methods)]
        for res in resolutions
    ]
    return _grid(["Resolution", *methods], rows)


def _metric_value(results: dict, method: str, scenario: str, obj: str, key: str) -> float | None:
    scenarios = results["methods"][method]["scenarios"]
    return scenarios.get(scenario, {}).get(obj, {}).get(key)


def _grid(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines
