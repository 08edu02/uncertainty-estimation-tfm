"""Run the minimum exploratory analysis required before modelling.

It reads the datasets downloaded by ``scripts/download_data.py``, writes one
JSON summary per scenario, a Markdown report and the EDA figures. Everything
here is regenerable with a single command, which is what the thesis requires.

Examples
--------
    uv run python scripts/run_eda.py
    uv run python scripts/run_eda.py --size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uq.data import credit, pneumonia  # noqa: E402
from uq.eda import figures, summary  # noqa: E402
from uq.settings import data_dir, ensure_dir, figures_dir, results_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=data_dir())
    parser.add_argument("--results-dir", type=Path, default=results_dir())
    parser.add_argument("--figures-dir", type=Path, default=figures_dir())
    parser.add_argument(
        "--size",
        type=int,
        default=pneumonia.DEFAULT_SIZE,
        choices=pneumonia.SUPPORTED_SIZES,
    )
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path


def _credit_report(payload: dict[str, Any]) -> list[str]:
    target = payload["target"]
    lines = [
        "### Escenario tabular (crédito)",
        "",
        f"- Filas: {payload['n_rows']:,} · variables: {payload['n_features']}",
        f"- Duplicados exactos (sin la etiqueta): {payload['n_duplicate_rows']:,}",
        f"- Valores faltantes: {payload['n_missing_values']:,}",
        f"- Tasa de impago: {target['positive_rate']:.4f} "
        f"({target['counts'].get('1', 0):,} de {payload['n_rows']:,})",
        "",
        "| Grupo | n | Tasa de impago |",
        "| --- | ---: | ---: |",
    ]
    for column, groups in payload["groups"].items():
        for group, stats in groups.items():
            lines.append(f"| {column} = {group} | {stats['n']:,} | {stats['positive_rate']:.4f} |")
    lines.append("")
    return lines


def _pneumonia_report(payload: dict[str, Any]) -> list[str]:
    lines = [
        "### Escenario de imágenes (PneumoniaMNIST)",
        "",
        "| Conjunto | Imágenes | Clase 0 | Clase 1 | Proporción clase 1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, stats in payload.items():
        lines.append(
            f"| {split} | {stats['n_samples']:,} | {stats['class_counts'].get('0', 0):,} "
            f"| {stats['class_counts'].get('1', 0):,} | {stats['class_rates'].get('1', 0):.4f} |"
        )
    first = next(iter(payload.values()))
    lines += [
        "",
        f"- Forma de las imágenes: {first['shape']} ({first['dtype']}), "
        f"píxeles en [{first['pixel_min']:.0f}, {first['pixel_max']:.0f}]",
        f"- Media de píxel: {first['pixel_mean']:.2f} ± {first['pixel_std']:.2f}",
        "",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = ensure_dir(args.results_dir / "eda")
    plots_dir = ensure_dir(args.figures_dir)

    # --- Tabular scenario ---------------------------------------------------
    frame = credit.load(args.data_dir)
    credit_summary = summary.tabular_summary(
        frame,
        credit.TARGET,
        sensitive=credit.SENSITIVE_COLUMNS[:3],
        categorical=credit.CATEGORICAL_COLUMNS,
    )
    _write_json(report_dir / "credit.json", credit_summary)
    figures.plot_credit_class_balance(frame, credit.TARGET, plots_dir / "credit_class_balance.png")
    figures.plot_credit_distributions(frame, plots_dir / "credit_distributions.png")
    print(
        f"[credit] {credit_summary['n_rows']:,} rows, "
        f"positive rate {credit_summary['target']['positive_rate']:.4f}, "
        f"{credit_summary['n_duplicate_rows']} duplicates"
    )

    # --- Image scenario -----------------------------------------------------
    pneumo_summary: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    samples = None
    for split in pneumonia.SPLITS:
        images, labels = pneumonia.load_split(split, args.data_dir, size=args.size)
        pneumo_summary[split] = summary.image_summary(images, labels, split=split)
        counts[split] = {
            str(key): int(value) for key, value in pneumo_summary[split]["class_counts"].items()
        }
        if split == "train":
            samples = (images, labels)
        print(f"[pneumonia] {split}: {len(labels):,} images")

    _write_json(report_dir / "pneumonia.json", pneumo_summary)
    figures.plot_split_balance(
        counts,
        plots_dir / "pneumonia_split_balance.png",
        title="PneumoniaMNIST class balance per split",
        class_names=pneumonia.CLASS_NAMES,
    )
    if samples is not None:
        figures.plot_image_samples(
            *samples,
            plots_dir / "pneumonia_samples.png",
            class_names=pneumonia.CLASS_NAMES,
        )

    lines = [
        "# EDA mínimo",
        "",
        "Generado por `scripts/run_eda.py`; no editar a mano.",
        "",
        *_credit_report(credit_summary),
        *_pneumonia_report(pneumo_summary),
        "## Figuras",
        "",
        "- `credit_class_balance.png`",
        "- `credit_distributions.png`",
        "- `pneumonia_split_balance.png`",
        "- `pneumonia_samples.png`",
        "",
    ]
    (report_dir / "report.md").write_text("\n".join(lines), "utf-8")
    print(f"[done] report written to {report_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
