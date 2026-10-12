"""Draw the 3,200-prompt sample (400 per task) from HALoGEN's prompt files.

Usage: python scripts/draw_sample.py
"""

import json
import subprocess
from pathlib import Path

import pandas as pd

SEED = 20260916
PER_TASK = 400

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "external" / "HALoGEN" / "prompts"
OUT_CSV = ROOT / "data" / "prompts.csv"
OUT_MANIFEST = ROOT / "data" / "prompts_manifest.json"

# task -> (source file, stratification column or None); Coding is not used
TASKS = {
    "biographies": ("prompts_biographies.csv", None),
    "false_presuppositions": ("prompts_falsepresupposition.csv", "condition"),
    "historical_events": ("prompts_historicalevents.csv", None),
    "rationalization_binary": ("prompts_rationalization_binary.csv", "category"),
    "rationalization_numerical": ("prompts_rationalization_numerical.csv", "condition"),
    "scientific_attribution": ("prompts_references.csv", None),
    "simplification": ("prompts_simplification.csv", None),
    "summarization": ("prompts_summarization.csv", None),
}


def allocate(counts: pd.Series, total: int) -> dict:
    """Proportional allocation with largest-remainder rounding."""
    exact = counts / counts.sum() * total
    alloc = exact.astype(int)
    shortfall = total - alloc.sum()
    for key in (exact - alloc).sort_values(ascending=False).index[:shortfall]:
        alloc[key] += 1
    return alloc.to_dict()


def repair_double_encoding(value):
    """Repair text that was decoded as cp1252 and re-encoded as UTF-8.

    The historical-events file stores 15 names this way ("LachevardiÃ¨re").
    Correctly encoded text fails the round trip and is returned unchanged.
    """
    if not isinstance(value, str) or value.isascii():
        return value
    try:
        raw = b"".join(
            # latin-1 covers the five bytes cp1252 leaves undefined
            c.encode("cp1252") if c.encode("cp1252", "ignore") else c.encode("latin-1")
            for c in value
        )
        fixed = raw.decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return fixed


def draw_task(task: str, filename: str, strat_col: str | None) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(PROMPTS_DIR / filename)
    df = df.map(repair_double_encoding)
    # a Windows checkout with core.autocrlf adds CR inside multi-line prompts
    df = df.map(lambda v: v.replace("\r\n", "\n") if isinstance(v, str) else v)
    df["source_row"] = range(len(df))

    if strat_col is None:
        picked = df.sample(n=PER_TASK, random_state=SEED)
        strata = {"_random": PER_TASK}
    else:
        strata = allocate(df[strat_col].value_counts().sort_index(), PER_TASK)
        picked = pd.concat(
            df[df[strat_col] == value].sample(n=n, random_state=SEED)
            for value, n in strata.items()
        )

    picked = picked.sort_values("source_row")
    meta_cols = [c for c in df.columns if c not in ("prompt", "source_row")]

    out = pd.DataFrame({
        "prompt_id": [f"{task}-{r:05d}" for r in picked["source_row"]],
        "task": task,
        "source_file": filename,
        "source_row": picked["source_row"].to_numpy(),
        "prompt": picked["prompt"].to_numpy(),
        "task_meta": [
            json.dumps({c: row[c] for c in meta_cols}, default=str, ensure_ascii=False)
            for _, row in picked.iterrows()
        ],
    })
    return out, {"pool": len(df), "stratified_on": strat_col, "strata": strata}


def pinned_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "external" / "HALoGEN"), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    frames, manifest_tasks = [], {}
    for task, (filename, strat_col) in TASKS.items():
        frame, info = draw_task(task, filename, strat_col)
        frames.append(frame)
        manifest_tasks[task] = info

    prompts = pd.concat(frames, ignore_index=True)
    assert len(prompts) == PER_TASK * len(TASKS)
    assert prompts["prompt_id"].is_unique

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    prompts.to_csv(OUT_CSV, index=False, lineterminator="\n")

    manifest = {
        "seed": SEED,
        "per_task": PER_TASK,
        "total": len(prompts),
        "pool_excluding_coding": sum(t["pool"] for t in manifest_tasks.values()),
        "halogen_commit": pinned_commit(),
        "tasks": manifest_tasks,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {len(prompts)} prompts -> {OUT_CSV}")
    for task, info in manifest_tasks.items():
        print(f"  {task:26} pool={info['pool']:5}  {info['strata']}")


if __name__ == "__main__":
    main()
