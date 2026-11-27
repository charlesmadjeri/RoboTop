FIELDS = (
    ("time_to_lost", "Frames until declared lost"),
    ("reacquire", "Frames to re-acquire"),
    ("held_frames", "Frames published from memory"),
)


def render(motion: dict) -> str:
    methods = list(motion)
    scenarios = list(next(iter(motion.values()))) if methods else []
    sections = []
    for key, title in FIELDS:
        rows = ["| Sequence | " + " | ".join(methods) + " |", "|" + "---|" * (len(methods) + 1)]
        for scenario in scenarios:
            cells = [_cell(motion[m][scenario][key]) for m in methods]
            rows.append(f"| {scenario} | " + " | ".join(cells) + " |")
        sections.append(f"### {title}\n\n" + "\n".join(rows))
    return "\n\n".join(sections) + "\n"


def _cell(value) -> str:
    return "never" if value is None else str(value)
