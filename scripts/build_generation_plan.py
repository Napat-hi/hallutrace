"""The generation plan: one row per generation, with its seed and fixed batch.

vLLM's greedy output depends on batch composition for most subject models, so
every seed and batch is fixed here, before any GPU runs, and replayed exactly.

Usage: python scripts/build_generation_plan.py   (reads data/prompts.csv)
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modal_app import MODEL_REVISIONS  # noqa: E402

PROMPTS_CSV = ROOT / "data" / "prompts.csv"
OUT_DIR = ROOT / "data" / "generation_plan"

MODELS = ["olmo2-7b", "dclm-7b", "olmo17-7b", "redpajama-7b", "pythia-69b"]

K_SAMPLES = 5
BATCH_SIZE = 64

# As in HALoGEN: a fixed token budget, no stop sequences, the released prompt sent
# verbatim, the same for every model and task.
MAX_NEW_TOKENS = 128
STOP_SEQUENCES: list[str] = []
PROMPT_TEMPLATE = "{prompt}"

GREEDY = {"temperature": 0.0, "top_p": 1.0}
SAMPLED = {"temperature": 1.0, "top_p": 0.9}


def derive_seed(prompt_id: str, model: str, sample_index: int) -> int:
    """Seed from (prompt_id, model, sample_index); the integer is stored."""
    key = f"{prompt_id}|{model}|{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


def gen_config_hash(cfg: dict) -> str:
    """gen_config_hash: a hash of every setting that changes the output."""
    payload = json.dumps({
        **cfg,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stop": STOP_SEQUENCES,
        "prompt_template": PROMPT_TEMPLATE,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def plan_for_model(prompt_ids: list[str], model: str) -> pd.DataFrame:
    rows = []
    for sample_index in range(K_SAMPLES + 1):  # 0 = greedy, 1..5 = sampled
        cfg = GREEDY if sample_index == 0 else SAMPLED
        cfg_hash = gen_config_hash(cfg)
        for prompt_id in sorted(prompt_ids):
            rows.append({
                "generation_id": f"{model}|{prompt_id}|{sample_index}",
                "prompt_id": prompt_id,
                "model": model,
                "sample_index": sample_index,
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "seed": derive_seed(prompt_id, model, sample_index),
                "escalated": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "gen_config_hash": cfg_hash,
            })

    plan = pd.DataFrame(rows)
    plan["batch_index"] = plan.index // BATCH_SIZE
    plan["batch_id"] = [f"{model}-b{b:05d}" for b in plan["batch_index"]]
    plan["batch_size"] = BATCH_SIZE
    return plan


def main():
    prompt_ids = pd.read_csv(PROMPTS_CSV)["prompt_id"].tolist()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "k_samples": K_SAMPLES,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stop_sequences": STOP_SEQUENCES,
        "prompt_template": PROMPT_TEMPLATE,
        "greedy": GREEDY,
        "sampled": SAMPLED,
        # The Hugging Face commit of each model's weights, recorded here and on every
        # generation record but not part of gen_config_hash, which covers decoding.
        "model_revisions": {m: MODEL_REVISIONS[m] for m in MODELS},
        "models": {},
    }
    for model in MODELS:
        plan = plan_for_model(prompt_ids, model)
        assert plan["generation_id"].is_unique
        plan.to_csv(OUT_DIR / f"{model}.csv", index=False)
        summary["models"][model] = {
            "generations": len(plan),
            "batches": int(plan["batch_index"].nunique()),
        }
        print(f"{model:13} {len(plan):6} generations  {plan['batch_index'].nunique():4} batches")

    (OUT_DIR / "plan_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
