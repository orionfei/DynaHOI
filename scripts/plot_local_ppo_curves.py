import argparse
import html
import json
import math
from pathlib import Path


PANELS = [
    ("loss", ["loss"]),
    ("value_loss", ["value_loss"]),
    ("policy_loss", ["policy_loss"]),
    ("reward / success", ["reward_mean", "success_rate"]),
    ("approx_kl", ["approx_kl"]),
    ("ratio_mean", ["ratio_mean"]),
]

COLORS = {
    "loss": "#2563eb",
    "value_loss": "#7c3aed",
    "policy_loss": "#dc2626",
    "reward_mean": "#059669",
    "success_rate": "#ea580c",
    "approx_kl": "#0891b2",
    "ratio_mean": "#4b5563",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def polyline_points(
    updates: list[float],
    values: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-8)
    points = []
    for update, value in zip(updates, values):
        x = left + (update - x_min) / x_span * width
        y = top + height - (value - y_min) / y_span * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def write_svg(rows: list[dict], output_path: Path):
    updates = [finite_float(row.get("update")) for row in rows]
    valid_updates = [value for value in updates if value is not None]
    if not valid_updates:
        raise ValueError("No valid update values found.")

    x_min = min(valid_updates)
    x_max = max(valid_updates)
    panel_width = 520
    panel_height = 230
    margin_left = 54
    margin_right = 20
    margin_top = 42
    margin_bottom = 34
    gap_x = 34
    gap_y = 34
    cols = 2
    row_count = math.ceil(len(PANELS) / cols)
    svg_width = cols * panel_width + (cols - 1) * gap_x
    svg_height = row_count * panel_height + (row_count - 1) * gap_y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827} .axis{stroke:#d1d5db;stroke-width:1} .grid{stroke:#eef2f7;stroke-width:1} .label{font-size:12px;fill:#4b5563} .title{font-size:15px;font-weight:700}</style>',
    ]

    for idx, (title, keys) in enumerate(PANELS):
        col = idx % cols
        row_idx = idx // cols
        panel_x = col * (panel_width + gap_x)
        panel_y = row_idx * (panel_height + gap_y)
        left = panel_x + margin_left
        top = panel_y + margin_top
        chart_width = panel_width - margin_left - margin_right
        chart_height = panel_height - margin_top - margin_bottom

        series = {}
        all_values = []
        for key in keys:
            key_updates = []
            key_values = []
            for update, log_row in zip(updates, rows):
                value = finite_float(log_row.get(key))
                if update is None or value is None:
                    continue
                key_updates.append(update)
                key_values.append(value)
                all_values.append(value)
            series[key] = (key_updates, key_values)
        if not all_values:
            continue

        y_min = min(all_values)
        y_max = max(all_values)
        if abs(y_max - y_min) < 1e-8:
            y_min -= 1.0
            y_max += 1.0
        else:
            pad = 0.08 * (y_max - y_min)
            y_min -= pad
            y_max += pad

        parts.append(f'<text class="title" x="{panel_x}" y="{panel_y + 18}">{html.escape(title)}</text>')
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + chart_height * frac
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}"/>')
        parts.append(f'<text class="label" x="{panel_x}" y="{top + 4}">{y_max:.3g}</text>')
        parts.append(f'<text class="label" x="{panel_x}" y="{top + chart_height}">{y_min:.3g}</text>')
        parts.append(f'<text class="label" x="{left}" y="{top + chart_height + 24}">update {x_min:.0f}</text>')
        parts.append(f'<text class="label" text-anchor="end" x="{left + chart_width}" y="{top + chart_height + 24}">update {x_max:.0f}</text>')

        legend_x = panel_x + panel_width - margin_right - 120
        legend_y = panel_y + 18
        for legend_idx, key in enumerate(keys):
            color = COLORS.get(key, "#111827")
            y = legend_y + legend_idx * 16
            parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
            parts.append(f'<text class="label" x="{legend_x + 24}" y="{y + 4}">{html.escape(key)}</text>')

        for key, (key_updates, key_values) in series.items():
            points = polyline_points(
                key_updates,
                key_values,
                x_min,
                x_max,
                y_min,
                y_max,
                left,
                top,
                chart_width,
                chart_height,
            )
            if points:
                parts.append(
                    f'<polyline fill="none" stroke="{COLORS.get(key, "#111827")}" stroke-width="2.2" '
                    f'stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'
                )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_png(rows: list[dict], output_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipped PNG: {exc}")
        return False

    updates = [row["update"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (title, keys) in zip(axes.flatten(), PANELS):
        for key in keys:
            ax.plot(updates, [row.get(key) for row in rows], marker="o", linewidth=1.8, label=key)
        ax.set_title(title)
        ax.set_xlabel("update")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, default=Path("train_metrics.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.train_log.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.train_log)
    if not rows:
        raise ValueError(f"No rows found in {args.train_log}")

    svg_path = output_dir / "training_curves.svg"
    png_path = output_dir / "training_curves.png"
    write_svg(rows, svg_path)
    print(f"Wrote {svg_path}")
    if write_png(rows, png_path):
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
