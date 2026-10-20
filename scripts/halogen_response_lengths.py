"""Measure how long HALoGEN's released model responses are, per task.

HALoGEN did not publish the token limits it generated with, so its released
responses are the evidence for output length. Tokens are approximated as
words * 1.3. Needs external/HALoGEN/responses.

Usage: python scripts/halogen_response_lengths.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESPONSES = ROOT / "external" / "HALoGEN" / "responses"
OUT = ROOT / "results" / "halogen_response_lengths.csv"

# responses/ folder -> task name
TASK_DIRS = {
    "biographies": "biographies",
    "historical_events": "historical_events",
    "numerical_falsepresupposition": "false_presuppositions",
    "rationalization_binary": "rationalization_binary",
    "rationalization_numerical": "rationalization_numerical",
    "references": "scientific_attribution",
    "simplification": "simplification",
    "summarization": "summarization",
}

TOKENS_PER_WORD = 1.3
SENTENCE_END = tuple('.!?"\')]')


def truncation_by_family(per_file: list[dict]) -> pd.DataFrame:
    """A fixed token cap shows up as long responses that end mid-sentence."""
    df = pd.DataFrame(per_file)
    df["family"] = df["source_model"].str.replace(
        r"_(biographies|relationship|numerical_inconsistency|yesno_response|"
        r"numerical_response|references|simplification|summarization)$",
        "", regex=True,
    )
    return (
        df.groupby("family")
        .agg(max_words_median=("words_max", "median"),
             cut_midsentence_share_median=("near_max_cut_share", "median"))
        .sort_values("max_words_median")
    )


def main():
    rows = []
    for folder, task in TASK_DIRS.items():
        for csv in sorted((RESPONSES / folder).glob("*.csv")):
            df = pd.read_csv(csv)
            if "response" not in df.columns:
                continue
            text = df["response"].fillna("").astype(str)
            words = text.str.split().str.len()
            near_max = text[words >= words.max() * 0.85].str.rstrip()
            rows.append({
                "near_max_cut_share": round(float((~near_max.str.endswith(SENTENCE_END)).mean()), 3),
                "task": task,
                "source_model": csv.stem,
                "n": len(df),
                "words_p50": words.quantile(0.50),
                "words_p90": words.quantile(0.90),
                "words_p95": words.quantile(0.95),
                "words_p99": words.quantile(0.99),
                "words_max": words.max(),
            })

    per_model = pd.DataFrame(rows)
    per_task = per_model.groupby("task").agg(
        models=("source_model", "count"),
        words_p50=("words_p50", "median"),
        words_p95=("words_p95", "median"),
        words_p99=("words_p99", "median"),
        words_max=("words_max", "max"),
    )
    per_task["approx_tokens_p95"] = (per_task["words_p95"] * TOKENS_PER_WORD).round()
    per_task["approx_tokens_p99"] = (per_task["words_p99"] * TOKENS_PER_WORD).round()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    per_model.to_csv(OUT, index=False)
    per_task.to_csv(OUT.with_name("halogen_response_lengths_by_task.csv"))

    families = truncation_by_family(rows)
    families.to_csv(OUT.with_name("halogen_truncation_by_model_family.csv"))

    print(per_task.to_string())
    print()
    print(families.to_string())


if __name__ == "__main__":
    main()
