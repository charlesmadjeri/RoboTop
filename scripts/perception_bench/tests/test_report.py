from perception_bench.report import render

HEADINGS = [
    "### Cube recall",
    "### Cube p95 error (px)",
    "### Cube false positives",
    "### Box recall",
    "### Box false positives",
    "### Latency (ms per frame)",
]


def _metric(recall=None, p95=None, fp=None):
    return {
        "recall": recall,
        "median_px": None,
        "p95_px": p95,
        "false_positive": fp,
        "n_eval": 10,
        "n_absent": 0,
    }


def _results():
    return {
        "frames": {"dark": 10, "clean": 10},
        "methods": {
            "hough": {
                "scenarios": {
                    "clean": {"cube": _metric(0.94, 3.26, 0.0), "box": _metric(0.5, None, 0.1)},
                    "dark": {"cube": _metric(0.6, 7.04, 0.2), "box": _metric(None, None, None)},
                },
                "latency_ms": {"320x240": 4.26, "640x480": 12.0},
            },
            "blob": {
                "scenarios": {
                    "clean": {"cube": _metric(0.88, 2.0, 0.3), "box": _metric(0.9, None, 0.05)},
                    "dark": {"cube": _metric(0.7, 5.5, 0.1), "box": _metric(None, None, None)},
                },
                "latency_ms": {"320x240": 1.5, "640x480": 3.0},
            },
        },
    }


def _table(text, heading):
    lines = text.splitlines()
    start = lines.index(heading)
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not all(set(cell) <= {"-"} for cell in cells):
                rows.append(cells)
        elif rows:
            break
    return rows


def test_renders_all_six_headings_in_order():
    text = render(_results())
    positions = [text.index(heading) for heading in HEADINGS]
    assert positions == sorted(positions)


def test_percent_px_and_na_formatting():
    text = render(_results())
    assert _table(text, "### Cube recall")[2] == ["clean", "94%", "88%"]
    assert _table(text, "### Cube p95 error (px)")[2] == ["clean", "3.3", "2.0"]
    assert _table(text, "### Box recall")[1] == ["dark", "n/a", "n/a"]


def test_recall_worst_row_is_minimum():
    rows = _table(render(_results()), "### Cube recall")
    assert rows[-1] == ["worst", "60%", "70%"]


def test_false_positive_worst_row_is_maximum():
    rows = _table(render(_results()), "### Cube false positives")
    assert rows[-1] == ["worst", "20%", "30%"]


def test_p95_worst_row_is_maximum():
    rows = _table(render(_results()), "### Cube p95 error (px)")
    assert rows[-1] == ["worst", "7.0", "5.5"]


def test_all_na_column_shows_na_in_worst_row():
    rows = _table(render(_results()), "### Box recall")
    assert rows[-1] == ["worst", "50%", "90%"]
    results = _results()
    results["methods"]["hough"]["scenarios"]["clean"]["box"]["recall"] = None
    results["methods"]["blob"]["scenarios"]["clean"]["box"]["recall"] = None
    rows = _table(render(results), "### Box recall")
    assert rows[-1] == ["worst", "n/a", "n/a"]


def test_column_order_follows_methods_and_row_order_follows_frames():
    rows = _table(render(_results()), "### Cube recall")
    assert rows[0] == ["Scenario", "hough", "blob"]
    assert [row[0] for row in rows[1:]] == ["dark", "clean", "worst"]


def test_latency_table_has_one_row_per_resolution():
    rows = _table(render(_results()), "### Latency (ms per frame)")
    assert rows == [
        ["Resolution", "hough", "blob"],
        ["320x240", "4.3", "1.5"],
        ["640x480", "12.0", "3.0"],
    ]


def test_frame_counts_are_listed_per_scenario():
    text = render(_results())
    assert "### Frames per scenario" in text
    assert "| dark | 10 |" in text
