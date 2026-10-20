"""HalluTrace's Modal app: model download, generation, verification.

Every stage is an entrypoint, run as `modal run modal_app.py::<name>`, for example
    modal run modal_app.py::download_models
"""

import os

import modal

app = modal.App("hallutrace")

# ---------------------------------------------------------------- volumes
models_vol = modal.Volume.from_name("hallutrace-models", create_if_missing=True)
wiki_vol = modal.Volume.from_name("hallutrace-wiki", create_if_missing=True)
data_vol = modal.Volume.from_name("hallutrace-data", create_if_missing=True)

MODELS_DIR = "/models"
WIKI_DIR = "/wiki"
DATA_DIR = "/data"

# ---------------------------------------------------------------- secrets
# Modal secrets "together" (TOGETHER_API_KEY), "huggingface" (HF_TOKEN) and
# "semanticscholar" (S2_API_KEY). Other names for the last two can be set in
# HF_SECRET_NAME / S2_SECRET_NAME, in the environment or in .modal-secrets.local
# next to this file (KEY=VALUE per line, read at import time).


def _local_secret_overrides() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".modal-secrets.local")
    out = {}
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


_SECRET_OVERRIDES = _local_secret_overrides()


def _secret_name(env_var: str, default: str) -> str:
    return os.environ.get(env_var) or _SECRET_OVERRIDES.get(env_var) or default


hf_secret = modal.Secret.from_name(_secret_name("HF_SECRET_NAME", "huggingface"))
together_secret = modal.Secret.from_name("together")
s2_secret = modal.Secret.from_name(_secret_name("S2_SECRET_NAME", "semanticscholar"))

# ---------------------------------------------------------------- subject models
SUBJECT_MODELS = {
    "olmo2-7b": "allenai/OLMo-2-1124-7B",
    "dclm-7b": "apple/DCLM-7B",
    "olmo17-7b": "allenai/OLMo-7B-0424-hf",
    "redpajama-7b": "togethercomputer/RedPajama-INCITE-7B-Base",
    "pythia-69b": "EleutherAI/pythia-6.9b",
}

# Hugging Face commit of the weights on hallutrace-models, keyed by volume name.
# fetch_model downloads these revisions; the generators refuse to run if the
# volume holds another, and stamp the value on every record as model_revision.
# data/generation_plan/plan_manifest.json carries the same five.
MODEL_REVISIONS = {
    "olmo2-7b": "7df9a82518afdecae4e8c026b27adccc8c1f0032",
    "dclm-7b": "c85bfa168f999ce27e954808bc005a2748fda5c5",
    "olmo17-7b": "e3c82a0871580b424d74e68acee866bb007df7df",
    "redpajama-7b": "78f7e482443971f4873ba3239f0ac810a367833b",
    "pythia-69b": "c0e3eee36dc47af0c49f361c74cfe459c09f7f23",
    "llama2-tokenizer": "01c7f73d771dfac7d292323805ebc428287df4f9",
    "gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
}

# Llama-2 tokenizer, for the v4_*_llama infini-gram indexes (a gated repo)
TOKENIZER_REPO = "meta-llama/Llama-2-7b-hf"

# ---------------------------------------------------------------- images
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]==0.35.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# the four models vLLM can serve
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    # transformers 5.x dropped all_special_tokens_extended, which vLLM 0.11.0 calls.
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "pandas==2.3.3")
    # vLLM 0.11.0 ignores this; kept so the image, and its smoke tests, stay unchanged
    .env({"VLLM_BATCH_INVARIANT": "1"})
)

# DCLM-7B only. Its config declares model_type "openlm" with no auto_map, so
# neither vLLM's registry nor its Transformers backend can load it
# (smoke_dclm_vllm), and open_lm's pins conflict with vLLM's.
OPEN_LM_COMMIT = "9bb92ef1689333534b7057942a20d18a46d1fa52"

# open_lm's requirements.txt pulls ray[all], mosaicml, wandb, etc. with no pins.
# Inference needs none of that, so install it --no-deps on a pinned stack where
# xformers matches torch. transformers stays pre-4.50 so the custom OpenLM model
# class still inherits generate().
openlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "xformers==0.0.28.post1",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        # pinned to what the image resolved to (locks/openlm_image.txt)
        "safetensors==0.8.0",
        "huggingface_hub==0.36.2",
        "tiktoken==0.14.0",
        "pandas==3.0.5",
    )
    .pip_install(
        f"git+https://github.com/mlfoundations/open_lm.git@{OPEN_LM_COMMIT}",
        extra_options="--no-deps",
    )
)


@app.function(
    image=download_image,
    volumes={MODELS_DIR: models_vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def fetch_model(item: tuple[str, str]) -> str:
    from huggingface_hub import snapshot_download

    name, repo_id = item
    # Only tokenizer files for Llama-2; one weight format for the rest.
    if repo_id == TOKENIZER_REPO:
        patterns = ["tokenizer*", "special_tokens_map.json", "*.model", "config.json"]
        ignore = None
    else:
        patterns = None
        ignore = ["pytorch_model*.bin"] if name != "redpajama-7b" else None
    snapshot_download(
        repo_id,
        revision=MODEL_REVISIONS[name],  # KeyError on purpose: pin before downloading
        local_dir=f"{MODELS_DIR}/{name}",
        allow_patterns=patterns,
        ignore_patterns=ignore,
    )
    models_vol.commit()
    return name


@app.function(
    image=download_image,
    volumes={MODELS_DIR: models_vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def list_models() -> list[str]:
    import os

    models_vol.reload()
    out = []
    for root, _, files in os.walk(MODELS_DIR):
        size = sum(os.path.getsize(os.path.join(root, f)) for f in files)
        if size:
            out.append(f"{root}: {size / 1e9:.1f} GB")
    return out


@app.function(image=download_image, volumes={MODELS_DIR: models_vol}, timeout=600)
def inventory() -> dict:
    import os

    models_vol.reload()
    out = {}
    for name in sorted(os.listdir(MODELS_DIR)):
        root = f"{MODELS_DIR}/{name}"
        files = []
        for dirpath, _, fnames in os.walk(root):
            if ".cache" in dirpath:
                continue
            for f in fnames:
                p = os.path.join(dirpath, f)
                files.append((os.path.relpath(p, root), os.path.getsize(p)))
        out[name] = sorted(files)
    return out


@app.function(image=download_image, volumes={MODELS_DIR: models_vol}, timeout=600)
def prune(dry_run: bool = True) -> list[str]:
    import glob
    import json
    import os

    models_vol.reload()
    doomed = []

    # Llama-2 is only needed for its tokenizer (infini-gram v4_*_llama indexes).
    tok = f"{MODELS_DIR}/llama2-tokenizer"
    doomed += glob.glob(f"{tok}/*.safetensors") + glob.glob(f"{tok}/pytorch_model*.bin")

    # Drop .bin shards only where a complete safetensors copy exists.
    for model_dir in glob.glob(f"{MODELS_DIR}/*"):
        if model_dir == tok:
            continue
        index = f"{model_dir}/model.safetensors.index.json"
        bins = glob.glob(f"{model_dir}/pytorch_model*.bin")
        if not bins or not os.path.exists(index):
            continue
        shards = set(json.load(open(index))["weight_map"].values())
        if all(os.path.exists(f"{model_dir}/{s}") for s in shards):
            doomed += bins

    freed = sum(os.path.getsize(p) for p in doomed)
    lines = [f"{'DRY RUN' if dry_run else 'DELETED'}: {p}" for p in doomed]
    lines.append(f"total: {freed / 1e9:.1f} GB")

    if not dry_run:
        for p in doomed:
            os.remove(p)
        models_vol.commit()
    return lines


SMOKE_PROMPTS = [
    "Tell me a bio of Ada Lovelace.",
    "The capital of France is",
    "Question: Is 97 a prime number? Answer:",
    "Write a short summary of the water cycle.",
]


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={MODELS_DIR: models_vol},
    timeout=30 * 60,
)
def smoke_dclm_vllm() -> dict:
    """Can vLLM serve DCLM-7B through its Transformers backend?"""
    import json
    import time
    import traceback

    path = f"{MODELS_DIR}/dclm-7b"
    cfg = json.load(open(f"{path}/config.json"))
    report = {
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "auto_map": cfg.get("auto_map"),
    }

    try:
        from vllm import LLM, SamplingParams

        t0 = time.time()
        llm = LLM(
            model=path,
            model_impl="transformers",
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,
            seed=0,
        )
        report["load_seconds"] = round(time.time() - t0, 1)

        params = SamplingParams(temperature=0.0, max_tokens=128)
        t0 = time.time()
        outs = llm.generate(SMOKE_PROMPTS, params)
        elapsed = time.time() - t0
        n_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

        report["ok"] = True
        report["tokens_per_second"] = round(n_tokens / elapsed, 1)
        report["samples"] = [o.outputs[0].text[:200] for o in outs]
    except Exception:
        report["ok"] = False
        report["error"] = traceback.format_exc()[-3000:]
    return report


@app.local_entrypoint()
def smoke_vllm():
    import json

    print(json.dumps(smoke_dclm_vllm.remote(), indent=2, ensure_ascii=False))


VLLM_MODELS = ["olmo2-7b", "olmo17-7b", "redpajama-7b", "pythia-69b"]

# Unrelated prompts used only to change batch composition around SMOKE_PROMPTS.
FILLER_PROMPTS = [f"Write a sentence about the number {i}." for i in range(60)]


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={MODELS_DIR: models_vol},
    timeout=30 * 60,
)
def smoke_vllm_model(name: str) -> dict:
    """Smoke test for a vLLM-served model: load, greedy twice, batch-composition check."""
    import time
    import traceback

    report = {"model": name, "image": "vllm_image", "gpu": "H100"}
    try:
        import vllm
        import vllm.envs as envs
        from vllm import LLM, SamplingParams

        report["vllm_version"] = vllm.__version__
        report["batch_invariant_env_known"] = hasattr(envs, "VLLM_BATCH_INVARIANT")

        t0 = time.time()
        llm = LLM(
            model=f"{MODELS_DIR}/{name}",
            dtype="bfloat16",
            max_model_len=2048,
            seed=0,
        )
        report["load_seconds"] = round(time.time() - t0, 1)

        params = SamplingParams(temperature=0.0, max_tokens=128)

        def gen(prompts):
            t0 = time.time()
            outs = llm.generate(prompts, params, use_tqdm=False)
            n = sum(len(o.outputs[0].token_ids) for o in outs)
            return [o.outputs[0].text for o in outs], n / (time.time() - t0)

        alone_1, _ = gen(SMOKE_PROMPTS)
        alone_2, _ = gen(SMOKE_PROMPTS)
        mixed, tps = gen(FILLER_PROMPTS[:30] + SMOKE_PROMPTS + FILLER_PROMPTS[30:])
        mixed_smoke = mixed[30:30 + len(SMOKE_PROMPTS)]

        report["ok"] = True
        report["tokens_per_second_batch64"] = round(tps, 1)
        report["greedy_identical_across_runs"] = alone_1 == alone_2
        report["identical_alone_vs_in_batch"] = alone_1 == mixed_smoke
        report["mismatched_prompts"] = [
            SMOKE_PROMPTS[i] for i, (a, b) in enumerate(zip(alone_1, mixed_smoke)) if a != b
        ]
        report["samples"] = [t[:200] for t in alone_1]
    except Exception:
        report["ok"] = False
        report["error"] = traceback.format_exc()[-3000:]
    return report


@app.local_entrypoint()
def smoke_vllm_all():
    import json
    from datetime import date
    from pathlib import Path

    out_dir = Path("results/smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    for report in smoke_vllm_model.map(VLLM_MODELS):
        report["run_date"] = date.today().isoformat()
        (out_dir / f"{report['model']}_vllm.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        keys = ("ok", "load_seconds", "tokens_per_second_batch64",
                "greedy_identical_across_runs", "identical_alone_vs_in_batch")
        print(report["model"], {k: report.get(k) for k in keys})
        if not report["ok"]:
            print(report["error"][-800:])


@app.function(image=openlm_image, timeout=10 * 60)
def openlm_import_check() -> str:
    """CPU-only: does `open_lm.hf` import and register the openlm model type?"""
    import traceback

    try:
        import open_lm.hf  # noqa: F401  registers model_type "openlm" with transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        return f"ok: openlm registered = {'openlm' in CONFIG_MAPPING}"
    except Exception:
        return traceback.format_exc()[-3000:]


@app.function(
    image=openlm_image,
    gpu="H100",
    volumes={MODELS_DIR: models_vol},
    timeout=30 * 60,
)
def smoke_dclm_openlm() -> dict:
    """Smoke test: load DCLM-7B via open_lm, greedy-generate twice, compare."""
    import time
    import traceback

    report = {}
    try:
        import open_lm.hf  # noqa: F401
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = f"{MODELS_DIR}/dclm-7b"
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()
        report["load_seconds"] = round(time.time() - t0, 1)

        def run():
            texts, n_new = [], 0
            t0 = time.time()
            for prompt in SMOKE_PROMPTS:
                ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=128, do_sample=False)
                new = out[0, ids.shape[1]:]
                n_new += new.numel()
                texts.append(tok.decode(new, skip_special_tokens=True))
            return texts, n_new / (time.time() - t0)

        first, tps = run()
        second, _ = run()

        report["ok"] = True
        report["tokens_per_second_unbatched"] = round(tps, 1)
        report["greedy_identical_across_runs"] = first == second
        report["samples"] = [t[:200] for t in first]
    except Exception:
        report["ok"] = False
        report["error"] = traceback.format_exc()[-3000:]
    return report


@app.local_entrypoint()
def check_openlm_import():
    print(openlm_import_check.remote())


@app.local_entrypoint()
def smoke_openlm():
    import json

    print(json.dumps(smoke_dclm_openlm.remote(), indent=2, ensure_ascii=False))


@app.local_entrypoint()
def prune_preview():
    print("\n".join(prune.remote(dry_run=True)))


@app.local_entrypoint()
def prune_apply():
    print("\n".join(prune.remote(dry_run=False)))


@app.local_entrypoint()
def show_inventory():
    for name, files in inventory.remote().items():
        print(f"\n== {name}")
        for rel, size in files:
            print(f"   {size / 1e9:7.2f} GB  {rel}")


@app.local_entrypoint()
def download_models():
    targets = list(SUBJECT_MODELS.items()) + [("llama2-tokenizer", TOKENIZER_REPO)]
    for name in fetch_model.map(targets):
        print(f"done: {name}")


@app.local_entrypoint()
def check():
    for line in list_models.remote():
        print(line)


# ================================================================ generation
# Reads data/generation_plan/{model}.csv and data/prompts.csv, runs the plan batch
# by batch in plan order, and writes one JSONL file per batch to the data volume.
# A batch whose file already exists is skipped, so a crashed run resumes.

from pathlib import Path  # noqa: E402

LOCAL_DATA = Path(__file__).parent / "data"
IMAGE_DATA = "/root/data"
# data/corpus_cache/ holds the infini-gram window caches: corpus text, gitignored, never uploaded.
DATA_IGNORE = ["corpus_cache", "corpus_cache/**"]

vllm_gen_image = vllm_image.add_local_dir(LOCAL_DATA, IMAGE_DATA, ignore=DATA_IGNORE)
openlm_gen_image = openlm_image.add_local_dir(LOCAL_DATA, IMAGE_DATA, ignore=DATA_IGNORE)


def _load_plan(model: str, max_batches: int, only_batches: str = ""):
    import pandas as pd

    plan = pd.read_csv(f"{IMAGE_DATA}/generation_plan/{model}.csv")
    prompts = pd.read_csv(f"{IMAGE_DATA}/prompts.csv").set_index("prompt_id")["prompt"]
    plan["prompt"] = plan["prompt_id"].map(prompts)
    assert plan["prompt"].notna().all(), "plan references a prompt_id missing from prompts.csv"
    if only_batches:
        wanted = {int(b) for b in only_batches.split(",")}
        plan = plan[plan["batch_index"].isin(wanted)]
    batches = [g for _, g in plan.groupby("batch_index", sort=True)]
    return batches[:max_batches] if max_batches else batches


def pilot_batch_indices(sample_indices=(0, 1)) -> str:
    """The plan batch holding the middle prompt of every task, per sample_index."""
    import pandas as pd

    plan = pd.read_csv(LOCAL_DATA / "generation_plan" / "olmo2-7b.csv")
    plan["task"] = plan["prompt_id"].str.rsplit("-", n=1).str[0]
    picked = set()
    for s in sample_indices:
        for _, g in plan[plan["sample_index"] == s].groupby("task"):
            picked.add(int(g["batch_index"].iloc[len(g) // 2]))
    return ",".join(str(b) for b in sorted(picked))


def _write_batch(path: str, records: list[dict]) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _checked_revision(model: str) -> str:
    """The HF commit on the volume for this model, which must equal MODEL_REVISIONS."""
    path = f"{MODELS_DIR}/{model}/.cache/huggingface/download/config.json.metadata"
    on_volume = open(path, encoding="utf-8").readline().strip()
    if on_volume != MODEL_REVISIONS[model]:
        raise RuntimeError(f"{model}: volume holds {on_volume}, MODEL_REVISIONS pins "
                           f"{MODEL_REVISIONS[model]}. Refusing to generate from unpinned weights.")
    return on_volume


def _base_record(row, engine: dict) -> dict:
    return {
        "generation_id": row.generation_id,
        "prompt_id": row.prompt_id,
        "model": row.model,
        "sample_index": int(row.sample_index),
        "temperature": float(row.temperature),
        "top_p": float(row.top_p),
        "seed": int(row.seed),
        "escalated": bool(row.escalated),
        "max_new_tokens": int(row.max_new_tokens),
        "gen_config_hash": row.gen_config_hash,
        "batch_id": row.batch_id,
        **engine,
    }


@app.function(
    image=vllm_gen_image,
    gpu="H100",
    volumes={MODELS_DIR: models_vol, DATA_DIR: data_vol},
    timeout=12 * 60 * 60,
)
def generate_vllm(model: str, max_batches: int = 0, out: str = "generations", only_batches: str = "") -> dict:
    import json
    import os
    import time
    from datetime import datetime, timezone

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    path = f"{MODELS_DIR}/{model}"
    revision = _checked_revision(model)  # before the model load, so a mismatch fails fast
    ctx = json.load(open(f"{path}/config.json")).get("max_position_embeddings", 2048)
    llm = LLM(model=path, dtype="bfloat16", max_model_len=ctx, seed=0)
    tok = llm.get_tokenizer()
    engine = {"engine": f"vllm-{vllm.__version__}", "gpu": torch.cuda.get_device_name(0),
              "model_revision": revision}

    stats = {"model": model, "context": ctx, "batches_done": 0, "batches_skipped": 0,
             "generations": 0, "prompt_too_long": 0, "finish_length": 0, "seconds": 0.0}
    t_start = time.time()

    for batch in _load_plan(model, max_batches, only_batches):
        out_path = f"{DATA_DIR}/{out}/{model}/{batch.batch_id.iloc[0]}.jsonl"
        if os.path.exists(out_path):
            stats["batches_skipped"] += 1
            continue

        rows = list(batch.itertuples(index=False))
        runnable, records = [], {}
        for r in rows:
            n_prompt = len(tok(r.prompt).input_ids)
            rec = _base_record(r, engine) | {"n_prompt_tokens": n_prompt}
            if n_prompt + r.max_new_tokens > ctx:
                rec |= {"response": None, "n_output_tokens": 0, "finish_reason": "prompt_too_long"}
                stats["prompt_too_long"] += 1
            else:
                runnable.append(r)
            records[r.generation_id] = rec

        if runnable:
            params = [
                SamplingParams(temperature=r.temperature, top_p=r.top_p,
                               max_tokens=int(r.max_new_tokens), seed=int(r.seed))
                for r in runnable
            ]
            outs = llm.generate([r.prompt for r in runnable], params, use_tqdm=False)
            now = datetime.now(timezone.utc).isoformat()
            for r, o in zip(runnable, outs):
                c = o.outputs[0]
                records[r.generation_id] |= {
                    "response": c.text,
                    "n_output_tokens": len(c.token_ids),
                    "finish_reason": c.finish_reason,
                    "generated_at": now,
                }
                stats["finish_length"] += c.finish_reason == "length"

        _write_batch(out_path, [records[r.generation_id] for r in rows])
        data_vol.commit()
        stats["batches_done"] += 1
        stats["generations"] += len(rows)

    stats["seconds"] = round(time.time() - t_start, 1)
    return stats


@app.function(
    image=openlm_gen_image,
    gpu="H100",
    volumes={MODELS_DIR: models_vol, DATA_DIR: data_vol},
    timeout=24 * 60 * 60,
)
def generate_openlm(
    model: str = "dclm-7b",
    max_batches: int = 0,
    out: str = "generations",
    shard: int = 0,
    n_shards: int = 1,
    only_batches: str = "",
) -> dict:
    """DCLM-7B via open_lm + HF generate, one prompt at a time.

    open_lm ignores the attention mask, so left-padded batches corrupt every
    shorter prompt (check_openlm_padding: 7 of 8 outputs changed). Generating one
    prompt at a time avoids padding, lets every sample use its own planned seed,
    and makes output independent of batch composition. Throughput comes from
    running n_shards containers; plan batches are kept only as output files.
    """
    import os
    import time
    from datetime import datetime, timezone

    import open_lm.hf  # noqa: F401
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = f"{MODELS_DIR}/{model}"
    revision = _checked_revision(model)  # before the model load, so a mismatch fails fast
    ctx = 2048
    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()
    engine = {"engine": f"open_lm-{OPEN_LM_COMMIT[:7]}+transformers-{transformers.__version__}",
              "gpu": torch.cuda.get_device_name(0), "model_revision": revision}

    stats = {"model": model, "shard": shard, "context": ctx, "batches_done": 0,
             "batches_skipped": 0, "generations": 0, "prompt_too_long": 0,
             "finish_length": 0, "seconds": 0.0}
    t_start = time.time()

    for i, batch in enumerate(_load_plan(model, max_batches, only_batches)):
        if i % n_shards != shard:
            continue
        out_path = f"{DATA_DIR}/{out}/{model}/{batch.batch_id.iloc[0]}.jsonl"
        if os.path.exists(out_path):
            stats["batches_skipped"] += 1
            continue

        rows = list(batch.itertuples(index=False))
        runnable, records = [], {}
        for r in rows:
            n_prompt = len(tok(r.prompt).input_ids)
            rec = _base_record(r, engine) | {"n_prompt_tokens": n_prompt}
            if n_prompt + r.max_new_tokens > ctx:
                rec |= {"response": None, "n_output_tokens": 0, "finish_reason": "prompt_too_long"}
                stats["prompt_too_long"] += 1
            else:
                runnable.append(r)
            records[r.generation_id] = rec

        for r in runnable:
            do_sample = r.temperature > 0
            kwargs = {"max_new_tokens": int(r.max_new_tokens), "do_sample": do_sample,
                      "pad_token_id": tok.eos_token_id}
            if do_sample:
                kwargs |= {"temperature": float(r.temperature), "top_p": float(r.top_p)}
            torch.manual_seed(int(r.seed))
            ids_in = tok(r.prompt, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                gen = lm.generate(ids_in, **kwargs)
            ids = gen[0, ids_in.shape[1]:].tolist()
            if tok.eos_token_id in ids:
                ids = ids[: ids.index(tok.eos_token_id)]
                reason = "stop"
            else:
                reason = "length"
            records[r.generation_id] |= {
                "response": tok.decode(ids, skip_special_tokens=True),
                "n_output_tokens": len(ids),
                "finish_reason": reason,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            stats["finish_length"] += reason == "length"

        _write_batch(out_path, [records[r.generation_id] for r in rows])
        data_vol.commit()
        stats["batches_done"] += 1
        stats["generations"] += len(rows)

    stats["seconds"] = round(time.time() - t_start, 1)
    return stats


@app.function(
    image=openlm_gen_image,
    gpu="H100",
    volumes={MODELS_DIR: models_vol},
    timeout=30 * 60,
)
def check_openlm_padding(n: int = 8) -> dict:
    """Does left padding change DCLM's greedy output? Compares batched vs one-by-one."""
    import open_lm.hf  # noqa: F401
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = f"{MODELS_DIR}/dclm-7b"
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()

    prompts = pd.read_csv(f"{IMAGE_DATA}/prompts.csv").groupby("task").head(1)["prompt"].tolist()[:n]
    prompts = [p for p in prompts if len(tok(p).input_ids) < 1800]

    def greedy(batch):
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = lm.generate(enc.input_ids, attention_mask=enc.attention_mask,
                            max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(x, skip_special_tokens=True) for x in g[:, enc.input_ids.shape[1]:]]

    batched = greedy(prompts)
    single = [greedy([p])[0] for p in prompts]
    lengths = [len(tok(p).input_ids) for p in prompts]
    return {
        "n_prompts": len(prompts),
        "prompt_token_lengths": lengths,
        "identical": [a == b for a, b in zip(batched, single)],
        "examples": [{"single": s[:120], "batched": b[:120]} for s, b in zip(single, batched)][:3],
    }


@app.local_entrypoint()
def generate(
    models: str = "all",
    max_batches: int = 0,
    out: str = "generations",
    dclm_shards: int = 8,
    batches: str = "",
):
    """Run the generation plan.

    Full run:  modal run modal_app.py::generate
    Pilot:     modal run modal_app.py::generate --batches pilot --out pilot_2
               (--batches pilot = every task's middle batch, greedy and sampled)
    """
    import json

    if batches == "pilot":
        batches = pilot_batch_indices()
        print("pilot batches:", batches)

    names = list(SUBJECT_MODELS) if models == "all" else models.split(",")
    calls = []
    for n in names:
        if n == "dclm-7b":
            calls += [generate_openlm.spawn(n, max_batches, out, s, dclm_shards, batches)
                      for s in range(dclm_shards)]
        else:
            calls.append(generate_vllm.spawn(n, max_batches, out, batches))
    for call in calls:
        print(json.dumps(call.get()))


token_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.22.1", "pandas==2.3.3")
    .add_local_dir(LOCAL_DATA, IMAGE_DATA, ignore=DATA_IGNORE)
)


@app.function(image=token_image, volumes={MODELS_DIR: models_vol}, timeout=30 * 60)
def prompt_length_report(max_new_tokens: int = 128) -> list[dict]:
    """CPU only: per model and task, how many prompts leave no room for the output."""
    import json

    import pandas as pd
    from tokenizers import Tokenizer

    prompts = pd.read_csv(f"{IMAGE_DATA}/prompts.csv")
    rows = []
    for model in SUBJECT_MODELS:
        cfg = json.load(open(f"{MODELS_DIR}/{model}/config.json"))
        ctx = 2048 if model == "dclm-7b" else cfg.get("max_position_embeddings", 2048)
        tok = Tokenizer.from_file(f"{MODELS_DIR}/{model}/tokenizer.json")
        n = pd.Series([len(e.ids) for e in tok.encode_batch(prompts["prompt"].tolist())])
        too_long = n + max_new_tokens > ctx
        for task, idx in prompts.groupby("task").groups.items():
            rows.append({
                "model": model, "context": ctx, "task": task,
                "prompts": len(idx),
                "tokens_p50": int(n[idx].median()), "tokens_max": int(n[idx].max()),
                "too_long": int(too_long[idx].sum()),
            })
    return rows


@app.local_entrypoint()
def prompt_lengths():
    import pandas as pd

    df = pd.DataFrame(prompt_length_report.remote())
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/prompt_lengths.csv", index=False)
    print(df[df.too_long > 0].to_string(index=False) if (df.too_long > 0).any() else "no prompt exceeds any context")
    print(df.groupby("model")[["too_long"]].sum().to_string())


api_image = modal.Image.debian_slim(python_version="3.12").pip_install("together==1.5.26", "requests==2.34.2")

TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


@app.function(image=api_image, secrets=[together_secret, s2_secret], timeout=5 * 60)
def check_api_keys() -> dict:
    """Check the Together and Semantic Scholar keys. Reports status only, never key values."""
    import os
    import time
    import traceback

    import requests

    report = {}

    key = os.environ.get("TOGETHER_API_KEY", "")
    report["together_key_set"] = bool(key) and key.lower() != "placeholder"
    try:
        from together import Together

        t0 = time.time()
        r = Together(api_key=key).chat.completions.create(
            model=TOGETHER_MODEL,
            messages=[{"role": "user", "content": "Is 97 a prime number? Answer yes or no."}],
            max_tokens=5,
            temperature=0.0,
        )
        report["together_ok"] = True
        report["together_model"] = r.model
        report["together_answer"] = r.choices[0].message.content.strip()
        report["together_seconds"] = round(time.time() - t0, 2)
    except Exception as e:
        report["together_ok"] = False
        report["together_error"] = f"{type(e).__name__}: {str(e)[:300]}"

    key = os.environ.get("S2_API_KEY", "")
    report["s2_key_set"] = bool(key) and key.lower() != "placeholder"
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": "Attention is all you need", "fields": "title", "limit": 1},
            headers={"x-api-key": key} if report["s2_key_set"] else {},
            timeout=30,
        )
        report["s2_status"] = resp.status_code
        report["s2_ok"] = resp.status_code == 200
        if resp.ok and resp.json().get("data"):
            report["s2_title"] = resp.json()["data"][0]["title"]
        elif not resp.ok:
            report["s2_error"] = resp.text[:300]
    except Exception:
        report["s2_ok"] = False
        report["s2_error"] = traceback.format_exc()[-300:]
    return report


@app.local_entrypoint()
def api_keys():
    import json

    print(json.dumps(check_api_keys.remote(), indent=2, ensure_ascii=False))


@app.local_entrypoint()
def padding_check():
    import json

    print(json.dumps(check_openlm_padding.remote(), indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- verification
# The CPU image that runs HALoGEN's verifiers through hallutrace/verifier_runner.py.
# Biographies needs a GPU and the Wikipedia index, and has its own image (bio_image).
#
# - PYTHONPATH must hold the repo root: the patched scorers/abstention_detector.py
#   and src/scientific_attribution.py import hallutrace.abstention.
# - TorchDynamo is off, as OpenFActScore's README asks for Gemma 3.
# - add_local_dir with copy=False adds files at container start, after the build,
#   so patches/ uses copy=True (git apply reads it during the build) and
#   hallutrace/ the default, so a code change does not rebuild the image.

HALOGEN_REPO = "https://github.com/AbhilashaRavichander/HALoGEN.git"
HALOGEN_COMMIT = "8b7dd77e9602af0b4c9d1430e60f474ac1b4eaa4"
HALOGEN_DIR = "/halogen"

verify_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "together==1.5.26",  # same pin as api_image
        "pandas==2.3.3",     # same pin as vllm_image
        "requests==2.34.2",
        "spacy==3.8.2",      # rationalization_binary loads en_core_web_sm
        "nltk==3.10.3",      # biographies calls nltk.download('punkt')
        # imported at module level by verifiers/get_model_responses.py; never called
        "openai==1.109.1",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    # copy=True: git apply below runs at build time and needs these files.
    .add_local_dir("patches", remote_path="/patches", copy=True)
    .run_commands(
        f"git clone --filter=blob:none {HALOGEN_REPO} {HALOGEN_DIR}",
        f"git -C {HALOGEN_DIR} checkout {HALOGEN_COMMIT}",
        f"git -C {HALOGEN_DIR} apply /patches/halogen-fixes.patch",
        f"git -C {HALOGEN_DIR} apply /patches/halogen-openfactscore.patch",
    )
    .env(
        {
            # verifier_runner puts HALOGEN_ROOT and its src/ on sys.path itself,
            # so /repo is the only entry this needs.
            "PYTHONPATH": "/repo",
            "HALOGEN_ROOT": HALOGEN_DIR,
            # OpenFActScore README: Gemma 3 is unstable without these.
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "TRANSFORMERS_NO_TORCH_COMPILE": "1",
        }
    )
    # MUST BE LAST. env() and run_commands() are build steps, and Modal refuses
    # any build step after an add_local_* with copy=False. copy=False is what we
    # want here: the files arrive at container startup, so editing our own code
    # does not rebuild the image.
    .add_local_dir("hallutrace", remote_path="/repo/hallutrace")
)


@app.function(image=verify_image, timeout=5 * 60)
def check_verify_image() -> dict:
    """Proves the image can import what it has to, before anything is spent.

    No key, no network, no GPU. It only checks that the patched HALoGEN and our
    package can be imported together, which is exactly what PYTHONPATH is for.
    """
    import ast
    import os
    import sys

    report = {}

    from hallutrace.abstention import PHRASES_FIXED
    from hallutrace.verifier_runner import TASKS, describe

    report["phrases"] = len(PHRASES_FIXED)
    report["tasks"] = len(TASKS)

    root = os.environ["HALOGEN_ROOT"]
    for path in (root, os.path.join(root, "src")):
        if path not in sys.path:
            sys.path.insert(0, path)

    # the patched files must import (PYTHONPATH) and hold no inline phrase list
    inline = {}
    for rel in ("scorers/abstention_detector.py", "src/scientific_attribution.py"):
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        tree = ast.parse(src)
        inline[rel] = sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, ast.List)
            and len(n.elts) > 10
            and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in n.elts)
        )
    report["inline_phrase_lists"] = inline

    import spacy

    spacy.load("en_core_web_sm")
    report["spacy"] = "en_core_web_sm loaded"

    report["describe_lines"] = len(describe().splitlines())
    return report


@app.local_entrypoint()
def verify_image_check():
    out = check_verify_image.remote()
    print(out)
    assert out["phrases"] == 30, out
    assert out["tasks"] == 9, out
    assert all(v == 0 for v in out["inline_phrase_lists"].values()), out
    print("verify_image is wired correctly")


# ---------------------------------------------------------------- FActScore knowledge source
# The Google Drive files FActScore's download_data.py fetches, downloaded once into
# hallutrace-wiki with their sha256 in MANIFEST.json. The Biographies image points
# HALLUTRACE_FACTSCORE_DIR at FACTSCORE_DIR. data.zip (the FActScore paper's labelled
# data) is not needed: the knowledge source is the .db alone.
FACTSCORE_DIR = f"{WIKI_DIR}/factscore"
FACTSCORE_FILES = {
    "demos.zip": "1IseEAflk1qqV0z64eM60Fs3dTgnbgiyt",
    "enwiki-20230401.db": "1mekls6OGOKLmt7gYtHs0WGf5oTamTNat",
}

wiki_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unzip", "curl")
)


@app.function(image=wiki_image, volumes={WIKI_DIR: wiki_vol}, timeout=6 * 60 * 60)
def fetch_factscore_data() -> dict:
    """Download FActScore's demos and Wikipedia 2023-04-01 index into hallutrace-wiki.

    Resumable: a file already on the volume is not fetched again. Zips are unpacked
    and deleted; sha256 is recorded for every file that lands.
    """
    import hashlib
    import json
    import os
    import subprocess
    import time
    from datetime import datetime, timezone

    os.makedirs(FACTSCORE_DIR, exist_ok=True)
    manifest_path = f"{FACTSCORE_DIR}/MANIFEST.json"
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()

    for name, file_id in FACTSCORE_FILES.items():
        if name in manifest:
            print(f"[skip] {name} already recorded")
            continue
        t = time.time()
        local = f"{FACTSCORE_DIR}/{name}.part"
        # Files over Drive's virus-scan limit (the 20 GB .db) return a warning page
        # that gdown cannot get past; confirm=t on drive.usercontent skips it.
        subprocess.run(["curl", "-fL", "--retry", "5", "-o", local,
                        f"https://drive.usercontent.google.com/download?id={file_id}"
                        "&export=download&confirm=t"], check=True)
        if os.path.getsize(local) < 1 << 20 and open(local, "rb").read(15).lower().startswith(b"<!doctype html"):
            import re
            page = open(local, encoding="utf-8", errors="replace").read()
            text = re.sub(r"<[^>]+>", " ", re.sub(r"(?s)<(script|style).*?</\1>", "", page))
            raise RuntimeError(f"Drive returned an HTML page for {name} ({file_id}), not the file: "
                               f"{' '.join(text.split())[:400]}")
        entry = {"drive_id": file_id, "bytes": os.path.getsize(local), "sha256": sha256(local),
                 "fetched_at": datetime.now(timezone.utc).isoformat(),
                 "seconds": round(time.time() - t, 1)}
        if name.endswith(".zip"):
            listing = subprocess.run(["unzip", "-Z1", local], capture_output=True, text=True,
                                     check=True).stdout.split()
            subprocess.run(["unzip", "-o", "-q", local, "-d", FACTSCORE_DIR], check=True)
            entry["unpacked"] = listing
            os.remove(local)
        else:
            os.replace(local, f"{FACTSCORE_DIR}/{name}")
        manifest[name] = entry
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        wiki_vol.commit()
        print(f"[done] {name}: {entry['bytes'] / 1e9:.2f} GB in {entry['seconds']} s")

    return {k: {"bytes": v["bytes"], "sha256": v["sha256"]} for k, v in manifest.items()}


@app.function(image=wiki_image, volumes={WIKI_DIR: wiki_vol}, timeout=10 * 60)
def check_factscore_data() -> dict:
    """Cheap read-only check that the index opens as SQLite: schema and one row, no full scan."""
    import json
    import sqlite3

    con = sqlite3.connect(f"file:{FACTSCORE_DIR}/enwiki-20230401.db?mode=ro", uri=True)
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    out = {"tables": {t: [r[1] for r in con.execute(f"pragma table_info({t})")] for t in tables},
           "indexes": [r[0] for r in con.execute("select name from sqlite_master where type='index'")]}
    row = con.execute(f"select * from {tables[0]} limit 1 offset 100000").fetchone()
    out["row_100000"] = [str(x)[:150] for x in row] if row else None
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------- lockfiles
# Several images pin only their top-level packages, so transitive versions float
# on rebuild. freeze_images records what each image actually resolved to, one
# file per image under locks/, so a later rebuild can be diffed against it.
# CPU only: pip freeze needs no GPU, even in the vLLM and open_lm images.
LOCKED_IMAGES = {
    "download_image": download_image,
    "vllm_image": vllm_image,
    "openlm_image": openlm_image,
    "token_image": token_image,
    "api_image": api_image,
    "verify_image": verify_image,
    "wiki_image": wiki_image,
}


@app.local_entrypoint()
def freeze_images():
    """modal run modal_app.py::freeze_images  ->  locks/<image>.txt

    Runs pip freeze in a Sandbox per image (a plain command, so the 3.11 open_lm
    image works too).
    """
    import pathlib

    out = pathlib.Path(__file__).parent / "locks"
    out.mkdir(exist_ok=True)
    cmd = "python --version && python -m pip freeze --all"
    boxes = {name: modal.Sandbox.create("bash", "-c", cmd, image=img, app=app, timeout=30 * 60)
             for name, img in LOCKED_IMAGES.items()}
    for name, sb in boxes.items():
        sb.wait()
        text = sb.stdout.read()
        if sb.returncode != 0:
            raise RuntimeError(f"{name}: pip freeze failed: {sb.stderr.read()[:500]}")
        (out / f"{name}.txt").write_text("# " + text, encoding="utf-8")
        print(f"{name}: {len(text.splitlines()) - 1} packages")


# ---------------------------------------------------------------- Biographies image
# GPU, OpenFActScore, both volumes; the same HALoGEN checkout and patches as
# verify_image.
#
# - OpenFActScore is installed --no-deps from a pinned commit: its pyproject asks
#   for bitsandbytes and pysqlite-binary, neither of which this configuration
#   uses. What it does import is listed explicitly below. openai is imported at
#   module level by factscorer.py even when no OpenAI model is used.
# - AFG and AFV point at the local Gemma copy on hallutrace-models, so a run never
#   depends on Hugging Face being reachable or the gated licence still holding.
# - HALLUTRACE_FACTSCORE_DIR is FActScore's data_dir, model_dir AND cache_dir at
#   once (the patch passes one path for all three), and it writes caches there.
#   /wiki/factscore is the only copy of the index outside Drive, so the verifier
#   gets its own directory holding symlinks to the index and demos instead.
# - HF_HOME on the models volume keeps the GTR retrieval encoder
#   (sentence-transformers/gtr-t5-large) from downloading on every container.
OPENFACTSCORE_COMMIT = "35c08f1f6137726986da71f151f001366ca683db"
BIO_FACTSCORE_DIR = "/factscore"

bio_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",                  # same as vllm_image
        "transformers==4.57.1",          # same as vllm_image; Gemma 3 needs >= 4.50
        "sentence-transformers==3.4.1",
        "rank-bm25==0.2.2",
        "spacy==3.8.2",                  # same as verify_image
        "nltk==3.10.3",                  # same as verify_image
        "pandas==2.3.3",                 # same as verify_image
        "openai==1.109.1",
        "together==1.5.26",
        "requests==2.34.2",
    )
    .run_commands(
        "python -m spacy download en_core_web_sm",
        "python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab",
    )
    .pip_install(f"git+https://github.com/lflage/OpenFActScore.git@{OPENFACTSCORE_COMMIT}",
                 extra_options="--no-deps")
    .add_local_dir("patches", remote_path="/patches", copy=True)
    .run_commands(
        f"git clone --filter=blob:none {HALOGEN_REPO} {HALOGEN_DIR}",
        f"git -C {HALOGEN_DIR} checkout {HALOGEN_COMMIT}",
        f"git -C {HALOGEN_DIR} apply /patches/halogen-fixes.patch",
        f"git -C {HALOGEN_DIR} apply /patches/halogen-openfactscore.patch",
    )
    .env(
        {
            "PYTHONPATH": "/repo",
            "HALOGEN_ROOT": HALOGEN_DIR,
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "TRANSFORMERS_NO_TORCH_COMPILE": "1",
            "HALLUTRACE_AFG_MODEL": f"{MODELS_DIR}/gemma-3-4b-it",
            "HALLUTRACE_AFV_MODEL": f"{MODELS_DIR}/gemma-3-4b-it",
            "HALLUTRACE_FACTSCORE_DIR": BIO_FACTSCORE_DIR,
            "HF_HOME": f"{MODELS_DIR}/.hf-cache",
            "NLTK_DATA": "/usr/share/nltk_data",
        }
    )
    .add_local_dir("hallutrace", remote_path="/repo/hallutrace")  # last, copy=False
)


def _bio_workdir():
    """A writable FActScore dir whose index and demos are symlinks into /wiki."""
    import os

    os.makedirs(BIO_FACTSCORE_DIR, exist_ok=True)
    for name in ("enwiki-20230401.db", "demos"):
        link = f"{BIO_FACTSCORE_DIR}/{name}"
        if not os.path.lexists(link):
            os.symlink(f"{FACTSCORE_DIR}/{name}", link)


def _halogen_on_path():
    import os
    import sys

    root = os.environ["HALOGEN_ROOT"]
    for path in (os.path.join(root, "src"), root):
        if path not in sys.path:
            sys.path.insert(0, path)


@app.function(image=bio_image, gpu="H100",
              volumes={MODELS_DIR: models_vol, WIKI_DIR: wiki_vol}, timeout=30 * 60)
def check_bio_image() -> dict:
    """Smoke test: imports, GPU, index, Gemma load, one retrieval. No scoring."""
    import json
    import time

    import torch

    _bio_workdir()
    _halogen_on_path()
    report = {"gpu": torch.cuda.get_device_name(0)}

    from verifiers.biographies import AFG_MODEL, AFV_MODEL, FACTSCORE_DIR as FS_DIR
    report["afg"], report["afv"], report["factscore_dir"] = AFG_MODEL, AFV_MODEL, FS_DIR

    from factscore.retrieval import DocDB, Retrieval
    db = DocDB(db_path=f"{FS_DIR}/enwiki-20230401.db")
    t = time.time()
    ret = Retrieval(db, f"{FS_DIR}/retrieval-smoke.json", f"{FS_DIR}/retrieval-smoke.pkl", batch_size=256)
    passages = ret.get_passages("Iggy Azalea", "Iggy Azalea is an Australian rapper.", k=5)
    report["retrieval_seconds"] = round(time.time() - t, 1)
    report["passage_titles"] = [p["title"] for p in passages]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t = time.time()
    tok = AutoTokenizer.from_pretrained(AFV_MODEL)
    lm = AutoModelForCausalLM.from_pretrained(AFV_MODEL).to("cuda")
    report["gemma_load_seconds"] = round(time.time() - t, 1)
    report["gemma_class"] = type(lm).__name__
    report["gpu_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)
    ids = tok("Iggy Azalea is an Australian rapper. True or False?\nAnswer:", return_tensors="pt").to("cuda")
    out = lm.generate(**ids, max_new_tokens=5, do_sample=False)
    report["gemma_says"] = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
    models_vol.commit()  # keeps the GTR encoder in HF_HOME
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


@app.function(image=bio_image, gpu="H100",
              volumes={MODELS_DIR: models_vol, WIKI_DIR: wiki_vol}, timeout=2 * 60 * 60)
def pilot_biographies(rows: list[dict]) -> dict:
    """Score a handful of responses end to end and measure time and GPU use."""
    import os
    import subprocess
    import threading
    import time

    import pandas as pd

    _bio_workdir()
    _halogen_on_path()
    from verifiers.biographies import BiographiesEvaluator

    samples, stop = [], threading.Event()

    def watch():
        while not stop.is_set():
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if q.returncode == 0:
                u, m = q.stdout.strip().split(", ")
                samples.append((int(u), int(m)))
            time.sleep(2)

    csv_path = "/tmp/bio_pilot.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    threading.Thread(target=watch, daemon=True).start()
    t = time.time()
    out_path = BiographiesEvaluator().evaluate(csv_path=csv_path, output_dir="/tmp/")
    seconds = time.time() - t
    stop.set()

    out = pd.read_csv(out_path) if isinstance(out_path, str) and os.path.exists(out_path) else None
    utils = [u for u, _ in samples]
    return {
        "responses": len(rows),
        "seconds": round(seconds, 1),
        "seconds_per_response": round(seconds / len(rows), 1),
        "gpu_util_mean": round(sum(utils) / len(utils), 1) if utils else None,
        "gpu_util_busy_share": round(sum(u > 10 for u in utils) / len(utils), 2) if utils else None,
        "gpu_mem_peak_mb": max((m for _, m in samples), default=None),
        "output_path": out_path,
        "output_rows": None if out is None else len(out),
        "output_head": None if out is None else out.head(15).to_dict("records"),
    }


@app.local_entrypoint()
def bio_pilot(per_model: int = 2, sample_index: int = 1):
    """modal run modal_app.py::bio_pilot  ->  results/bio_pilot.json

    Takes the first non-empty Biographies responses per model from results/pilot_2,
    each for a different person: biographies.py joins its output back to the input
    by name, so a repeated name raises InvalidIndexError after all scoring is done.
    """
    import glob
    import json
    import pathlib

    import pandas as pd

    here = pathlib.Path(__file__).parent
    names = {r.prompt_id: json.loads(r.task_meta)["name"]
             for r in pd.read_csv(here / "data/prompts.csv").itertuples() if r.task == "biographies"}
    rows, used = [], set()
    for model in SUBJECT_MODELS:
        picked = 0
        for path in sorted(glob.glob(str(here / f"results/pilot_2/{model}/*.jsonl"))):
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                if (rec["prompt_id"] in names and rec["sample_index"] == sample_index
                        and names[rec["prompt_id"]] not in used
                        and (rec["response"] or "").strip() and picked < per_model):
                    used.add(names[rec["prompt_id"]])
                    rows.append({"name": names[rec["prompt_id"]], "response": rec["response"],
                                 "generation_id": rec["generation_id"]})
                    picked += 1
    print(f"{len(rows)} responses")
    report = pilot_biographies.remote(rows)
    report["generation_ids"] = [r["generation_id"] for r in rows]
    (here / "results/bio_pilot.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "output_head"}, indent=2, default=str))


@app.function(image=bio_image, gpu="H100",
              volumes={MODELS_DIR: models_vol, WIKI_DIR: wiki_vol}, timeout=3 * 60 * 60)
def bio_throughput(groups: list[list[dict]]) -> list[dict]:
    """Score several groups of different people in one container, timing each.

    Every evaluate() builds a fresh FactScorer and loads Gemma twice (AFG, then AFV),
    so each group pays the same fixed overhead. The difference between two group
    sizes divided by the difference in responses is the marginal cost per response.
    """
    import os
    import subprocess
    import threading
    import time

    import pandas as pd

    _bio_workdir()
    _halogen_on_path()
    from verifiers.biographies import BiographiesEvaluator

    results = []
    for g, rows in enumerate(groups):
        samples, stop = [], threading.Event()

        def watch():
            while not stop.is_set():
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
                if q.returncode == 0:
                    u, m = q.stdout.strip().split(", ")
                    samples.append((int(u), int(m)))
                time.sleep(2)

        csv_path = f"/tmp/bio_group_{g}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        t = time.time()
        out_path = BiographiesEvaluator().evaluate(csv_path=csv_path, output_dir="/tmp/")
        seconds = time.time() - t
        stop.set()
        watcher.join()

        out = pd.read_csv(out_path)
        atoms = sum(str(a).count("\t") for a in out["atoms"]) if "atoms" in out else None
        utils = [u for u, _ in samples]
        results.append({
            "responses": len(rows),
            "seconds": round(seconds, 1),
            "atoms": atoms,
            "no_facts": int((out["atoms"].astype(str).str.contains("None\tNone")).sum()) if "atoms" in out else None,
            "gpu_util_mean": round(sum(utils) / len(utils), 1) if utils else None,
            "gpu_mem_peak_mb": max((m for _, m in samples), default=None),
        })
        print(results[-1])
    return results


@app.local_entrypoint()
def bio_scale(sizes: str = "10,50", sample_index: int = 1):
    """modal run modal_app.py::bio_scale  ->  results/bio_scale.json

    Round-robins across the five models so each group mixes all of them, and
    never repeats a person across or within groups.
    """
    import glob
    import json
    import pathlib

    import pandas as pd

    here = pathlib.Path(__file__).parent
    names = {r.prompt_id: json.loads(r.task_meta)["name"]
             for r in pd.read_csv(here / "data/prompts.csv").itertuples() if r.task == "biographies"}
    pools = {}
    for model in SUBJECT_MODELS:
        pools[model] = []
        for path in sorted(glob.glob(str(here / f"results/pilot_2/{model}/*.jsonl"))):
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                if (rec["prompt_id"] in names and rec["sample_index"] == sample_index
                        and (rec["response"] or "").strip()):
                    pools[model].append(rec)

    wanted = [int(s) for s in sizes.split(",")]
    used, stream = set(), []
    while len(stream) < sum(wanted) and any(pools.values()):
        for model in SUBJECT_MODELS:
            while pools[model]:
                rec = pools[model].pop(0)
                if names[rec["prompt_id"]] not in used:
                    used.add(names[rec["prompt_id"]])
                    stream.append({"name": names[rec["prompt_id"]], "response": rec["response"],
                                   "generation_id": rec["generation_id"]})
                    break
    if len(stream) < sum(wanted):
        raise SystemExit(f"only {len(stream)} distinct people available, asked for {sum(wanted)}")
    groups, i = [], 0
    for n in wanted:
        groups.append(stream[i:i + n])
        i += n

    results = bio_throughput.remote(groups)
    report = {"groups": results}
    if len(results) >= 2:
        a, b = results[0], results[-1]
        per = (b["seconds"] - a["seconds"]) / (b["responses"] - a["responses"])
        report["marginal_seconds_per_response"] = round(per, 2)
        report["fixed_overhead_seconds"] = round(a["seconds"] - per * a["responses"], 1)
        if a["atoms"] is not None and b["atoms"] != a["atoms"]:
            report["marginal_seconds_per_atom"] = round((b["seconds"] - a["seconds"]) / (b["atoms"] - a["atoms"]), 2)
    report["generation_ids"] = [[r["generation_id"] for r in g] for g in groups]
    (here / "results/bio_scale.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "generation_ids"}, indent=2))


# ---------------------------------------------------------------- backfill model_revision
def _with_revision(rec: dict) -> dict:
    """Return rec with model_revision placed right after gpu, as new records have it."""
    if "model_revision" in rec:
        if rec["model_revision"] != MODEL_REVISIONS[rec["model"]]:
            raise ValueError(f"{rec['generation_id']}: model_revision {rec['model_revision']} "
                             f"!= pinned {MODEL_REVISIONS[rec['model']]}")
        return rec
    out = {}
    for k, v in rec.items():
        out[k] = v
        if k == "gpu":
            out["model_revision"] = MODEL_REVISIONS[rec["model"]]
    out.setdefault("model_revision", MODEL_REVISIONS[rec["model"]])
    return out


def _backfill_jsonl(path: str) -> int:
    """Rewrite one batch file with model_revision on every record. Returns records changed."""
    import json

    lines = open(path, encoding="utf-8").read().splitlines()
    recs = [json.loads(line) for line in lines if line.strip()]
    new = [_with_revision(r) for r in recs]
    changed = sum("model_revision" not in r for r in recs)
    if changed:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return changed


@app.function(image=download_image, volumes={DATA_DIR: data_vol}, timeout=30 * 60)
def backfill_model_revision(outs: list[str]) -> dict:
    """Add model_revision to generation records already on hallutrace-data."""
    import glob

    data_vol.reload()
    report = {}
    for out in outs:
        files = sorted(glob.glob(f"{DATA_DIR}/{out}/*/*.jsonl"))
        report[out] = {"files": len(files), "records_changed": sum(_backfill_jsonl(p) for p in files)}
    data_vol.commit()
    print(report)
    return report


@app.local_entrypoint()
def backfill_revisions(outs: str = "pilot_1,pilot_2"):
    """modal run modal_app.py::backfill_revisions  (volume and local results/ copies)"""
    import glob
    import pathlib

    here = pathlib.Path(__file__).parent
    local = {}
    for out in outs.split(","):
        files = sorted(glob.glob(str(here / "results" / out / "*" / "*.jsonl")))
        local[out] = {"files": len(files), "records_changed": sum(_backfill_jsonl(p) for p in files)}
    print("local results/:", local)
    print("volume hallutrace-data:", backfill_model_revision.remote(outs.split(",")))


# ---------------------------------------------------------------- before generating: pinned revisions on the volume
@app.function(image=download_image, volumes={MODELS_DIR: models_vol}, timeout=10 * 60)
def check_revisions() -> dict:
    """CPU only. Runs the same _checked_revision the generators call, for every pinned model."""
    out = {}
    for name in MODEL_REVISIONS:
        try:
            out[name] = _checked_revision(name)[:7] + " ok"
        except Exception as e:  # report every model, not just the first failure
            out[name] = f"FAIL {type(e).__name__}: {e}"
    print(out)
    return out


# ---------------------------------------------------------------- Biographies verification
# One call scores one Biographies CSV on hallutrace-data through
# hallutrace.verifier_runner, which runs it once per (model, sample_index). Every
# call gets its own FActScore cache directory, so parallel calls never write the
# same pickle.
#
# The cache lives on hallutrace-data under factscore_cache/<input stem>, not in the
# container, so a call that dies half way resumes from FActScore's own caches
# (atomic facts, retrieval, validator answers) instead of starting over. The output
# file is the resume marker: if it exists the call returns without scoring.
BIO_CACHE_ROOT = f"{DATA_DIR}/factscore_cache"


@app.function(image=bio_image, gpu="H100", timeout=24 * 60 * 60,
              volumes={MODELS_DIR: models_vol, WIKI_DIR: wiki_vol, DATA_DIR: data_vol})
def verify_biographies(input_rel: str, output_rel: str, force: bool = False, batch: bool = True) -> dict:
    """Score DATA_DIR/input_rel with OpenFActScore, write DATA_DIR/output_rel.

    input needs columns name, response, model, sample_index (extra columns are
    fine). Paths are relative to the hallutrace-data volume. batch=True (the
    default) sends Gemma's prompts in batches (hallutrace/bio_batch.py);
    batch=False is the released one-prompt-at-a-time path.
    """
    import os
    import time

    import pandas as pd

    data_vol.reload()
    src, dst = f"{DATA_DIR}/{input_rel}", f"{DATA_DIR}/{output_rel}"
    if os.path.exists(dst) and not force:
        return {"skipped": True, "output": output_rel}

    stem = os.path.splitext(os.path.basename(input_rel))[0]
    cache = f"{BIO_CACHE_ROOT}/{stem}"
    os.makedirs(cache, exist_ok=True)
    for name in ("enwiki-20230401.db", "demos"):
        link = f"{cache}/{name}"
        if not os.path.lexists(link):
            os.symlink(f"{FACTSCORE_DIR}/{name}", link)
    # biographies.py reads this at import; a reused container has it imported already,
    # with an earlier call's directory, so the module's copy is set too
    os.environ["HALLUTRACE_FACTSCORE_DIR"] = cache
    import sys as _sys
    if "verifiers.biographies" in _sys.modules:
        _sys.modules["verifiers.biographies"].FACTSCORE_DIR = cache

    from hallutrace.verifier_runner import run_verifier, shard_plan

    if batch:
        from hallutrace import bio_batch
        bio_batch.enable()
    plan = shard_plan("biographies", src)
    t = time.time()
    run_verifier("biographies", src, dst)
    seconds = time.time() - t
    data_vol.commit()

    out = pd.read_csv(dst)
    return {"skipped": False, "output": output_rel, "shards": len(plan),
            "rows_in": sum(p["rows"] for p in plan), "rows_out": len(out),
            "seconds": round(seconds, 1), "cache": cache, "batch": batch}


@app.local_entrypoint()
def bio_verify_test(people: int = 2, force: bool = False):
    """modal run modal_app.py::bio_verify_test  ->  results/bio_verify_test.json

    The same `people` names for every model, sampled pass, so the file repeats each
    name five times and only works because verify_biographies shards it.
    Run twice with --force to see how much the persistent cache saves.
    """
    import glob
    import json
    import pathlib

    import pandas as pd

    here = pathlib.Path(__file__).parent
    names = {r.prompt_id: json.loads(r.task_meta)["name"]
             for r in pd.read_csv(here / "data/prompts.csv").itertuples() if r.task == "biographies"}
    by_model = {}
    for model in SUBJECT_MODELS:
        by_model[model] = {}
        for path in sorted(glob.glob(str(here / f"results/pilot_2/{model}/*.jsonl"))):
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                if rec["prompt_id"] in names and rec["sample_index"] == 1 and (rec["response"] or "").strip():
                    by_model[model][rec["prompt_id"]] = rec
    shared = sorted(set.intersection(*(set(v) for v in by_model.values())))[:people]
    rows = [{"name": names[p], "response": by_model[m][p]["response"], "model": m,
             "sample_index": 1, "generation_id": by_model[m][p]["generation_id"]}
            for m in SUBJECT_MODELS for p in shared]
    upload_bio_input.remote("verifier_test/bio_in.csv", rows)
    report = verify_biographies.remote("verifier_test/bio_in.csv", "verifier_test/bio_out.csv", force)
    report["people"] = [names[p] for p in shared]
    (here / "results/bio_verify_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- Biographies: batched vs unbatched
def _bio_scale_rows(group: int = -1) -> list[dict]:
    """The fixed bio_scale set (results/bio_scale.json, last group = 50 responses), rebuilt
    from results/pilot_2 by generation_id so every check scores the same responses."""
    import glob
    import json
    import pathlib

    import pandas as pd

    here = pathlib.Path(__file__).parent
    ids = json.loads((here / "results/bio_scale.json").read_text(encoding="utf-8"))["generation_ids"][group]
    names = {r.prompt_id: json.loads(r.task_meta)["name"]
             for r in pd.read_csv(here / "data/prompts.csv").itertuples() if r.task == "biographies"}
    recs = {}
    for path in glob.glob(str(here / "results/pilot_2/*/*.jsonl")):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if rec["generation_id"] in ids:
                recs[rec["generation_id"]] = rec
    return [{"name": names[recs[g]["prompt_id"]], "response": recs[g]["response"], "generation_id": g}
            for g in ids]


@app.function(image=bio_image, volumes={WIKI_DIR: wiki_vol}, timeout=20 * 60)
def check_bio_batch_parity(rows: list[dict]) -> dict:
    """CPU, stub models: the batched code builds the same prompts and parses the same
    answers as the pinned OpenFActScore code. Costs cents; run before any GPU check."""
    from hallutrace import bio_batch

    out = bio_batch.parity_check(rows, f"{FACTSCORE_DIR}/demos")
    print(out)
    return out


@app.local_entrypoint()
def bio_batch_parity():
    """modal run modal_app.py::bio_batch_parity  (CPU, no Gemma)"""
    print(check_bio_batch_parity.remote(_bio_scale_rows()))


@app.function(image=bio_image, gpu="H100", timeout=3 * 60 * 60,
              volumes={MODELS_DIR: models_vol, WIKI_DIR: wiki_vol, DATA_DIR: data_vol})
def bio_batch_runs(rows: list[dict]) -> dict:
    """The five runs of the batching check, each with fresh Gemma caches:
    u1, u2 unbatched (the noise floor); b batched; b_afv batched validator on u1's facts;
    u_afv unbatched validator on u1's facts (the same-facts AFV floor).
    Retrieval caches are carried from run to run: retrieval is deterministic and is not
    what the check measures. Only u1 retrieves cold, so compare speed on u2 vs b.

    Each finished run is saved to hallutrace-data:bio_batch_check/ with the retrieval
    cache, so a rerun after a crash or a closed laptop skips the runs already done."""
    import glob
    import json
    import os
    import shutil
    import tempfile
    import time

    from factscore.factscorer import FactScorer

    from hallutrace import bio_batch

    gemma = f"{MODELS_DIR}/gemma-3-4b-it"
    topics, gens = [r["name"] for r in rows], [r["response"] for r in rows]
    saved_dir = f"{DATA_DIR}/bio_batch_check"
    shared = f"{saved_dir}/retrieval"
    os.makedirs(shared, exist_ok=True)
    ids = [r["generation_id"] for r in rows]

    def load(label):
        path = f"{saved_dir}/{label}.json"
        if not os.path.exists(path):
            return None
        res = json.load(open(path, encoding="utf-8"))
        assert res.pop("generation_ids") == ids, f"{path} was run on other responses"
        print(label, "loaded from the volume", {k: res[k] for k in ("seconds", "score")})
        return res

    def save(label, res):
        with open(f"{saved_dir}/{label}.json", "w", encoding="utf-8") as f:
            json.dump({**res, "generation_ids": ids}, f, ensure_ascii=False)
        data_vol.commit()

    def run(label, batched, atomic_facts=None):
        saved = load(label)
        if saved is not None:
            return saved
        d = tempfile.mkdtemp(prefix=f"{label}-")
        for name in ("enwiki-20230401.db", "demos"):
            os.symlink(f"{FACTSCORE_DIR}/{name}", f"{d}/{name}")
        for f in glob.glob(f"{shared}/retrieval-*"):
            shutil.copy(f, d)
        (bio_batch.enable if batched else bio_batch.disable)()
        scorer = FactScorer(afg_model=gemma, afv_model=gemma, is_npm=False,
                            data_dir=d, model_dir=d, cache_dir=d)
        t = time.time()
        out = scorer.get_score(topics, gens, gamma=10, atomic_facts=atomic_facts)
        seconds = round(time.time() - t, 1)
        for f in glob.glob(f"{d}/retrieval-*"):
            shutil.copy(f, shared)
        decisions = out["decisions"]
        res = {"seconds": seconds, "score": float(out["score"]),
               "facts": [None if x is None else [a["atom"] for a in x] for x in decisions],
               "verdicts": [None if x is None else [bool(a["is_supported"]) for a in x] for x in decisions]}
        print(label, {k: v for k, v in res.items() if k in ("seconds", "score")})
        save(label, res)
        return res

    runs = {"u1": run("u1", False)}
    runs["u2"] = run("u2", False)
    runs["b"] = run("b", True)
    runs["b_afv"] = run("b_afv", True, atomic_facts=runs["u1"]["facts"])
    # same-facts AFV floor: the unbatched validator on u1's facts
    runs["u_afv"] = run("u_afv", False, atomic_facts=runs["u1"]["facts"])
    return runs


def _verdict_agreement(facts_a, verdicts_a, facts_b, verdicts_b):
    """Per-fact agreement over facts present in both runs of the same response, paired by
    identical text in order of occurrence. Returns (agree, compared)."""
    agree = compared = 0
    for fa, va, fb, vb in zip(facts_a, verdicts_a, facts_b, verdicts_b):
        if fa is None or fb is None:
            continue
        pool = {}
        for f, v in zip(fb, vb):
            pool.setdefault(f, []).append(v)
        for f, v in zip(fa, va):
            if pool.get(f):
                compared += 1
                agree += v == pool[f].pop(0)
    return agree, compared


@app.local_entrypoint()
def bio_batch_check():
    """modal run modal_app.py::bio_batch_check  ->  results/bio_batch_check.json  (H100, ~$3-4)

    The pass rules are in the comments below. Runs already saved on
    hallutrace-data:bio_batch_check/ are loaded, not rerun.
    """
    import json
    import pathlib

    rows = _bio_scale_rows()
    runs = bio_batch_runs.remote(rows)
    u1, u2, b, bv = runs["u1"], runs["u2"], runs["b"], runs["b_afv"]

    def pairs(x, y):
        return [(p, q) for p, q in zip(x["facts"], y["facts"]) if p is not None and q is not None]

    def same_lists(x, y):
        ps = pairs(x, y)
        return sum(p == q for p, q in ps), len(ps)

    def overlap(x, y):
        """Mean per-response F1 of the two fact multisets, after whitespace normalisation."""
        import collections
        f1s = []
        for p, q in pairs(x, y):
            cp = collections.Counter(" ".join(f.split()) for f in p)
            cq = collections.Counter(" ".join(f.split()) for f in q)
            f1s.append(2 * sum((cp & cq).values()) / (len(p) + len(q)) if p or q else 1.0)
        return round(sum(f1s) / len(f1s), 4)

    def facts_per_response(x):
        lists = [p for p in x["facts"] if p is not None]
        return round(sum(map(len, lists)) / len(lists), 2)

    # AFG samples, so two unbatched runs share no identical fact list. Batching passes AFG
    # if its fact overlap with each unbatched run is within 0.05 of the unbatched-vs-
    # unbatched overlap, with as many facts per response (within 10% of the unbatched mean).
    afg_floor, afg_test = same_lists(u1, u2), same_lists(u1, b)
    ov_floor = overlap(u1, u2)
    ov_test = round((overlap(u1, b) + overlap(u2, b)) / 2, 4)
    n_unbatched = (facts_per_response(u1) + facts_per_response(u2)) / 2
    n_batched = facts_per_response(b)
    afv_floor = _verdict_agreement(u1["facts"], u1["verdicts"], u2["facts"], u2["verdicts"])
    afv_test = _verdict_agreement(u1["facts"], u1["verdicts"], bv["facts"], bv["verdicts"])
    rate = lambda t: t[0] / t[1] if t[1] else None
    afg_pass = ov_test >= ov_floor - 0.05 and abs(n_batched - n_unbatched) <= 0.10 * n_unbatched
    # AFV: the floor is the unbatched validator on u1's own facts (u_afv). Batching passes
    # if b_afv vs u1 is at least that floor, or at least 99%.
    uv = runs["u_afv"]
    afv_floor_same = _verdict_agreement(u1["facts"], u1["verdicts"], uv["facts"], uv["verdicts"])
    afv_pass = (rate(afv_test) or 0) >= 0.99 or (rate(afv_test) or 0) >= (rate(afv_floor_same) or 1)
    report = {
        "responses": len(rows),
        "afg": {"identical_lists": {"floor_u1_u2": afg_floor, "batched_u1_b": afg_test},
                "fact_overlap_f1": {"floor_u1_u2": ov_floor, "batched_mean_u1b_u2b": ov_test},
                "facts_per_response": {"unbatched_mean": round(n_unbatched, 2), "batched": n_batched},
                "pass": afg_pass},
        "afv_agreement": {"floor_u1_u2_matched_facts": [*afv_floor, rate(afv_floor)],
                          "floor_u1_uafv_same_facts": [*afv_floor_same, rate(afv_floor_same)],
                          "batched_u1_bafv": [*afv_test, rate(afv_test)], "pass": afv_pass},
        "pass": afg_pass and afv_pass,
        "seconds": {k: v["seconds"] for k, v in runs.items()},
        "score": {k: v["score"] for k, v in runs.items()},
        # u1 retrieves cold; u2 and b both reuse its retrieval cache.
        "speedup_u2_over_b": round(u2["seconds"] / b["seconds"], 2),
        "generation_ids": [r["generation_id"] for r in rows],
        "runs": runs,
    }
    here = pathlib.Path(__file__).parent
    (here / "results/bio_batch_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("runs", "generation_ids")}, indent=2))


# Batch-size sweep: does the number of Gemma prompts per padded
# generate call change verdicts or speed? Validator only, on u1's own 1,076 facts, so every
# setting scores the same facts and is compared with u1 like b_afv and u_afv were.
BIO_SIZES = [("cap1", "response", 1), ("cap4", "response", 4), ("cap8", "response", 8),
             ("response", "response", None), ("x32", "across", 32), ("x64", "across", 64),
             ("x96", "across", 96)]


@app.function(image=bio_image, gpu="H100", timeout=3 * 60 * 60,
              volumes={MODELS_DIR: models_vol, WIKI_DIR: wiki_vol, DATA_DIR: data_vol})
def bio_batch_size_runs(rows: list[dict]) -> dict:
    """Per setting: fresh Gemma cache; pass 1 records the validator prompts per response
    (retrieval, no Gemma); pass 2 generates them, timed, either response by response with at
    most `size` prompts per call, or pooled across responses in calls of `size`; pass 3 reruns
    scoring from the now complete cache. "response" is the setting the runs use."""
    import glob
    import json
    import os
    import shutil
    import gc
    import tempfile
    import time

    import torch
    from factscore.factscorer import FactScorer

    from hallutrace import bio_batch

    saved_dir = f"{DATA_DIR}/bio_batch_check"
    u1 = json.load(open(f"{saved_dir}/u1.json", encoding="utf-8"))
    ids = [r["generation_id"] for r in rows]
    assert u1["generation_ids"] == ids
    topics, gens, facts = [r["name"] for r in rows], [r["response"] for r in rows], u1["facts"]
    gemma = f"{MODELS_DIR}/gemma-3-4b-it"
    bio_batch.enable()

    def scorer_in(d):
        for name in ("enwiki-20230401.db", "demos"):
            os.symlink(f"{FACTSCORE_DIR}/{name}", f"{d}/{name}")
        for f in glob.glob(f"{saved_dir}/retrieval/retrieval-*"):
            shutil.copy(f, d)
        return FactScorer(afg_model=gemma, afv_model=gemma, is_npm=False,
                          data_dir=d, model_dir=d, cache_dir=d)

    out = {}
    for label, mode, size in BIO_SIZES:
        path = f"{saved_dir}/size_{label}.json"
        if os.path.exists(path):
            out[label] = json.load(open(path, encoding="utf-8"))
            print(label, "loaded from the volume")
            continue
        scorer = scorer_in(tempfile.mkdtemp(prefix=f"{label}-"))
        lm = scorer.lm

        per_response = []  # pass 1: prompts only, dummy answers never enter the cache
        real = lm.generate_many
        lm.generate_many = lambda ps, max_output_length=128: (per_response.append(list(ps)),
                                                              [("True", None)] * len(ps))[1]
        t = time.time()
        scorer.get_score(topics, gens, gamma=10, atomic_facts=facts)
        prompt_seconds = time.time() - t
        lm.generate_many = real

        if lm.model is None:
            lm.load_model()
        bio_batch.MAX_ROWS = size
        # the default budget for per-response settings; room for the pooled ones
        bio_batch.TOKEN_BUDGET = 64_000 if mode == "response" else 150_000
        t = time.time()
        try:
            if mode == "response":
                for ps in per_response:
                    lm.generate_many(ps)
            else:
                lm.generate_many([p for ps in per_response for p in ps])
        except torch.cuda.OutOfMemoryError:
            out[label] = {"mode": mode, "size": size, "error": "CUDA out of memory"}
            print(label, "out of memory")
            del scorer, lm
            gc.collect()
            torch.cuda.empty_cache()
            continue
        finally:
            bio_batch.MAX_ROWS, bio_batch.TOKEN_BUDGET = None, 64_000
        gemma_seconds = time.time() - t

        res = scorer.get_score(topics, gens, gamma=10, atomic_facts=facts)  # pass 3, all cached
        verdicts = [None if x is None else [bool(a["is_supported"]) for a in x] for x in res["decisions"]]
        out[label] = {"mode": mode, "size": size, "prompts": sum(map(len, per_response)),
                      "prompt_seconds": round(prompt_seconds, 1), "gemma_seconds": round(gemma_seconds, 1),
                      "score": float(res["score"]), "facts": facts, "verdicts": verdicts}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**out[label], "generation_ids": ids}, f)
        data_vol.commit()
        print(label, {k: out[label][k] for k in ("prompts", "prompt_seconds", "gemma_seconds", "score")})
        del scorer, lm
        gc.collect()
        torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def bio_batch_sizes():
    """modal run modal_app.py::bio_batch_sizes  ->  results/bio_batch_sizes.json  (H100, ~$2)

    Agreement with u1 on the same facts, per setting, next to the two figures already
    measured: u_afv (the released one-prompt path) and b_afv (per-response batching)."""
    import json
    import pathlib

    here = pathlib.Path(__file__).parent
    check = json.loads((here / "results/bio_batch_check.json").read_text(encoding="utf-8"))
    u1 = check["runs"]["u1"]
    rows = _bio_scale_rows()
    out = bio_batch_size_runs.remote(rows)
    rate = lambda a, n: round(a / n, 4) if n else None
    table = {}
    for label, r in [("released_one_at_a_time (u_afv)", check["runs"]["u_afv"]),
                     ("per_response (b_afv)", check["runs"]["b_afv"])] + list(out.items()):
        if "error" in r:
            table[label] = r
            continue
        agree, n = _verdict_agreement(u1["facts"], u1["verdicts"], r["facts"], r["verdicts"])
        table[label] = {"agree_with_u1": [agree, n, rate(agree, n)], "score": round(r["score"], 4),
                        **{k: r[k] for k in ("mode", "size", "gemma_seconds", "prompt_seconds") if k in r}}
    report = {"u1_score": round(u1["score"], 4), "settings": table, "runs": out}
    (here / "results/bio_batch_sizes.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(json.dumps(table, indent=2))


@app.function(image=download_image.pip_install("pandas==2.3.3"), volumes={DATA_DIR: data_vol}, timeout=600)
def upload_bio_input(rel: str, rows: list[dict]) -> str:
    import os

    import pandas as pd

    os.makedirs(os.path.dirname(f"{DATA_DIR}/{rel}"), exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{DATA_DIR}/{rel}", index=False)
    data_vol.commit()
    return rel


# ---------------------------------------------------------------- verification runs
# Runs HALoGEN's verifiers over the generation batches. hallutrace/verifier_inputs.py decides
# what is pending and builds each input; this puts it on the volume, calls the
# verifier and commits. A unit is done only once its output file exists, so a crash
# or a timeout leaves it pending for the next run.
#
#   /data/{gen}/{model}/{batch}.jsonl            the generations (gen = generations)
#   /data/verifier/{gen}/inputs/{model}/{unit}.csv  what the verifier read
#   /data/verifier/{gen}/inputs/{model}/{unit}.meta.csv  every generation, sent or not
#   /data/verifier/{gen}/outputs/{model}/{unit}.csv what the verifier wrote
#   /data/verifier/{gen}/reports/{time}.json        one report per run, and latest.json
#   /data/cache/api_cache.sqlite                 every Together answer
#
# Run it with --detach. run_verifiers spawns the run and polls it in short waits, so a
# dropped connection cannot cancel it; verifier_status reads the report at any time.
# One container, one unit at a time: the API cache is SQLite, so the run works on a
# local copy and writes it back to the volume after every unit.

verifier_run_image = verify_image.add_local_dir(LOCAL_DATA, IMAGE_DATA, ignore=DATA_IGNORE)


def _verifier_root(gen: str, prompt_version: str = "released") -> str:
    """Where one verification run writes; the v2 prompts get their own root."""
    root = f"{DATA_DIR}/verifier/{gen}"
    return root if prompt_version == "released" else f"{root}-revised-prompts"
API_CACHE_PATH = f"{DATA_DIR}/cache/api_cache.sqlite"


@app.function(image=verifier_run_image, secrets=[together_secret, s2_secret],
              volumes={DATA_DIR: data_vol}, timeout=24 * 60 * 60)
def verifier_run(gen: str = "generations", tasks: str = "", per_task: int = 0,
              dry_run: bool = False, prompt_version: str = "released",
              redo: str = "", redo_tag: str = "") -> dict:
    """Verify every pending unit under DATA_DIR/gen, one task at a time in turn.

    tasks     comma separated task names, empty for all eight
    per_task  verified units each task should end up with, counting those
              already done, 0 for no cap
    dry_run   build and write the inputs, report what would be sent, verify nothing
    prompt_version  "released" (HALoGEN's) or "v2": v2 swaps the
              extraction prompts in the LLM client and writes under its own output root,
              so released outputs are never mistaken for v2 ones or skipped
    redo      comma separated unit ids to verify again: their old output and
              inputs move to superseded/{redo_tag}/ first (verifier_inputs.supersede), so
              nothing is deleted. Not moved on a dry run.
    """
    import os
    import time
    import traceback

    from hallutrace import verifier_inputs as bx
    from hallutrace import cache
    from hallutrace.abstention import AbstentionDetector
    from hallutrace.llm_client import LLMClient, TogetherBackend
    from hallutrace.verifier_runner import run_verifier

    data_vol.reload()
    gen_root = f"{DATA_DIR}/{gen}"
    out_root = _verifier_root(gen, prompt_version)
    prompts = bx.load_prompts(f"{IMAGE_DATA}/prompts.csv")
    redo_ids = [u for u in redo.split(",") if u]
    superseded = []
    if redo_ids:
        if not redo_tag:
            raise ValueError("redo needs a redo_tag, the folder the old files move to")
        known = {u.unit_id: u for u in bx.discover(gen_root, f"{IMAGE_DATA}/generation_plan")}
        unknown = [u for u in redo_ids if u not in known]
        if unknown:
            raise ValueError("redo names units that do not exist: %s" % unknown[:5])
        if not dry_run:
            superseded = bx.supersede(out_root, [known[u] for u in redo_ids], redo_tag)
            data_vol.commit()
    pending = bx.pending(gen_root, out_root, f"{IMAGE_DATA}/generation_plan")
    wanted = [t for t in tasks.split(",") if t]
    done = bx.done_per_task(gen_root, out_root, f"{IMAGE_DATA}/generation_plan")
    done_m = bx.done_per_task(gen_root, out_root, f"{IMAGE_DATA}/generation_plan", by_model=True)
    order = bx.first_per_task(pending, wanted, per_task, done, done_m)

    import shutil

    local_cache = "/tmp/api_cache.sqlite"
    os.makedirs(os.path.dirname(API_CACHE_PATH), exist_ok=True)
    if os.path.exists(API_CACHE_PATH):
        shutil.copy(API_CACHE_PATH, local_cache)
    conn = cache.connect_cache(local_cache)
    cache.init_cache_db(conn)

    def save_cache() -> None:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        shutil.copy(local_cache, API_CACHE_PATH + ".tmp")
        os.replace(API_CACHE_PATH + ".tmp", API_CACHE_PATH)
    swapper = None
    if prompt_version == "v2":
        from hallutrace.prompt_swap import Swapper

        swapper = Swapper()
    elif prompt_version != "released":
        raise ValueError("prompt_version must be 'released' or 'v2', got %r" % prompt_version)
    client = LLMClient(TogetherBackend(os.environ["TOGETHER_API_KEY"]),
                       cache.TogetherCache(conn), rewrite=swapper)
    detector = AbstentionDetector("fixed")

    import json
    from datetime import datetime, timezone

    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = f"{out_root}/reports"
    os.makedirs(report_dir, exist_ok=True)

    report = {"gen": gen, "started": started, "pending": len(pending),
              "done_before": done, "selected": len(order),
              "dry_run": dry_run, "prompt_version": prompt_version,
              "redo": redo_ids, "superseded": superseded,
              "finished": False, "units": []}

    def save_report() -> None:
        body = json.dumps(report, indent=2, ensure_ascii=False)
        for name in (f"{started}.json", "latest.json"):
            with open(f"{report_dir}/{name}", "w", encoding="utf-8") as f:
                f.write(body)
    for unit in order:
        row = {"unit": unit.unit_id, "task": unit.task}
        try:
            prep = bx.prepare(unit, prompts, detector)
            inp, _ = bx.write_inputs(prep, out_root)
            row["outcomes"] = prep.counts()
            row["sent"] = len(prep.rows)
            out = bx.output_path(out_root, unit)
            if dry_run:
                report["units"].append(row)
                save_report()
                data_vol.commit()
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            before = client.stats()
            swaps_before = sum(swapper.swapped.values()) if swapper is not None else 0
            t = time.time()
            if not prep.rows:
                # Nothing could yield a unit; the sidecar already says why.
                open(out, "w").close()
            elif unit.task == "biographies":
                data_vol.commit()  # verify_biographies runs in its own container
                rel = lambda p: os.path.relpath(p, DATA_DIR)  # noqa: E731
                row["bio"] = verify_biographies.remote(rel(inp), rel(out))
                data_vol.reload()
            else:
                run_verifier(unit.task, inp, out, client=client)
            after = client.stats()
            row.update({k: after[k] - before[k] for k in after})
            row["seconds"] = round(time.time() - t, 1)
            row["done"] = os.path.exists(out)
            if row.get("failures"):
                # the verifiers turn a failed call into "", scored as hallucinated,
                # and still write output; move it aside so the next run retries
                row["superseded"] = bx.supersede(out_root, [unit], "failed/" + started)
                row["done"] = False
            if swapper is not None:
                from hallutrace.prompt_swap import swap_missing

                row["swapped"] = sum(swapper.swapped.values()) - swaps_before
                if not row.get("superseded") and swap_missing(
                        unit.task, row.get("calls", 0) + row.get("cache_hits", 0), row["swapped"]):
                    # released output under the v2 root: move it aside
                    row["swap_warning"] = "v2 run swapped fewer prompts than LLM calls"
                    row["superseded"] = bx.supersede(out_root, [unit], "noswap/" + started)
                    row["done"] = False
        except Exception as e:  # noqa: BLE001  one bad unit must not stop the rest
            row["error"] = "%s: %s" % (type(e).__name__, e)
            row["trace"] = traceback.format_exc()[-1500:]
        if not dry_run:
            save_cache()
        report["units"].append(row)
        report["client"] = client.stats()
        if swapper is not None:
            report["swap"] = swapper.stats()
        save_report()
        data_vol.commit()

    report["client"] = client.stats()
    if swapper is not None:
        report["swap"] = swapper.stats()
    report["failures"] = client.failures[:20]
    report["finished"] = True
    save_report()
    data_vol.commit()
    return report


def _print_verifier_report(report: dict, gen: str) -> None:
    import json
    import pathlib

    out = pathlib.Path(__file__).parent / f"results/verifier_{gen}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if not report.get("finished", True):
        print("STILL RUNNING or stopped early: report as of the last finished unit")
    for u in report["units"]:
        status = "ERROR " + u["error"] if "error" in u else (
            "dry run" if report["dry_run"] else "done" if u.get("done") else
            "WARNING v2 swapped fewer prompts than calls, moved aside" if u.get("swap_warning") else
            "%d failed calls, moved aside to retry" % u["failures"] if u.get("superseded")
            else "no output")
        print("%-40s %-26s sent %3s  %s" % (u["unit"], u["task"], u.get("sent", "?"), status))
    print("pending %d, selected %d, client %s" % (report["pending"], report["selected"], report.get("client")))
    print("wrote", out)


@app.local_entrypoint()
def run_verifiers(gen: str = "generations", tasks: str = "", per_task: int = 0,
              dry_run: bool = False, prompt_version: str = "released",
              redo: str = "", redo_tag: str = ""):
    """modal run --detach modal_app.py::run_verifiers --gen pilot_2 --per-task 1

    --redo takes comma separated unit ids or a path to a local file with one
    unit id per line, and needs --redo-tag; see verifier_run.

    Spawns the run, then waits for it a minute at a time, printing progress
    from the report on the volume. A dropped connection is retried rather than
    raised, so it cannot cancel the run. Prints a summary and writes
    results/verifier_{gen}.json at the end; verifier_status does the same later.
    """
    import time
    from datetime import datetime, timedelta, timezone

    # Reports are named by UTC start time; ignore any from before this run.
    since = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y%m%dT%H%M%SZ")
    if redo and os.path.isfile(redo):
        # A file of unit ids, one per line, read here on the local machine.
        with open(redo) as f:
            redo = ",".join(line.strip() for line in f if line.strip())
    call = verifier_run.spawn(gen, tasks, per_task, dry_run, prompt_version, redo, redo_tag)
    print("started", call.object_id, "(verifier_status reads its report any time)")
    errors = 0
    while True:
        try:
            report = call.get(timeout=60)
            break
        except Exception as e:  # noqa: BLE001  a timeout, a dropped connection, or the run failing
            name = type(e).__name__
            # A timeout is checked first: Python's TimeoutError is an OSError.
            timeout = "Timeout" in name
            dropped = not timeout and (isinstance(e, (ConnectionError, OSError)) or any(
                k in name for k in ("StreamTerminated", "Connection", "GRPC", "ClientClosed")))
            if not timeout and not dropped:
                # The run itself raised. Retrying only waits on a failure.
                print("verifier_run failed: %s: %s" % (name, e))
                raise
            if dropped:
                errors += 1
                print("  connection problem (%s), retrying %d/30" % (type(e).__name__, errors))
                if errors >= 30:
                    print("giving up waiting; the run carries on if started with --detach")
                    return
                time.sleep(10)
                continue
            errors = 0
            try:
                now = verifier_report.remote(gen, prompt_version)
                if now.get("started", "") >= since:
                    print("  %d of %d units done" % (len(now.get("units", [])), now.get("selected", 0)))
            except Exception:  # noqa: BLE001  progress is best effort
                pass
    _print_verifier_report(report, gen if prompt_version == "released" else "%s-revised-prompts" % gen)


@app.function(image=download_image, volumes={DATA_DIR: data_vol}, timeout=120)
def verifier_report(gen: str = "generations", prompt_version: str = "released") -> dict:
    import json
    import os

    data_vol.reload()
    path = f"{_verifier_root(gen, prompt_version)}/reports/latest.json"
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8"))


@app.local_entrypoint()
def verifier_status(gen: str = "generations", prompt_version: str = "released"):
    """modal run modal_app.py::verifier_status --gen pilot_2 --prompt-version v2

    The latest verifier_run report from the volume, finished or not.
    """
    report = verifier_report.remote(gen, prompt_version)
    if not report:
        print("no report yet for", gen)
        return
    _print_verifier_report(report, gen if prompt_version == "released" else "%s-revised-prompts" % gen)


# ---------------------------------------------------------------------------
# the full Biographies run, in parallel on the verification units
# ---------------------------------------------------------------------------

@app.function(image=verifier_run_image, volumes={DATA_DIR: data_vol}, timeout=24 * 60 * 60)
def bio_full_run(gen: str = "generations", per_call: int = 4, max_groups: int = 0,
                 dry_run: bool = False) -> dict:
    """Verify every pending Biographies unit, `per_call` units per verify_biographies call,
    all calls at once (Modal caps the workspace at 10 GPUs). hallutrace/bio_full.py says
    why and how. max_groups > 0 stops after that many groups (a test run); dry_run writes
    the inputs and reports the groups without calling Gemma.
    """
    import json
    import os
    from datetime import datetime, timezone

    from hallutrace import verifier_inputs as bx
    from hallutrace import bio_full as bf
    from hallutrace.abstention import AbstentionDetector

    data_vol.reload()
    gen_root, out_root = f"{DATA_DIR}/{gen}", f"{DATA_DIR}/verifier/{gen}"
    prompts = bx.load_prompts(f"{IMAGE_DATA}/prompts.csv")
    detector = AbstentionDetector("fixed")
    units = [u for u in bx.pending(gen_root, out_root, f"{IMAGE_DATA}/generation_plan")
             if u.task == bx.BIOGRAPHIES]
    groups = bf.group_units(units, per_call)
    if max_groups:
        groups = groups[:max_groups]

    jobs = []
    for g in groups:
        preps = [bx.prepare(u, prompts, detector) for u in g]
        for p in preps:
            bx.write_inputs(p, out_root)
        inp, out = bf.group_paths(out_root, g)
        jobs.append({"group": bf.group_id(g), "model": g[0].model, "sample_index": g[0].sample_index,
                     "units": len(g), "sent": bf.write_group_input(preps, inp),
                     "_preps": preps, "_inp": inp, "_out": out})
    data_vol.commit()

    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = f"{out_root}/reports"
    os.makedirs(report_dir, exist_ok=True)
    report = {"gen": gen, "started": started, "per_call": per_call, "pending_units": len(units),
              "groups": len(jobs), "responses": sum(j["sent"] for j in jobs),
              "dry_run": dry_run, "finished": False, "done": []}

    def save_report():
        body = json.dumps(report, indent=2, ensure_ascii=False)
        for name in (f"bio_full_{started}.json", "bio_full_latest.json"):
            with open(f"{report_dir}/{name}", "w", encoding="utf-8") as f:
                f.write(body)
        data_vol.commit()

    public = lambda j: {k: v for k, v in j.items() if not k.startswith("_")}  # noqa: E731
    if dry_run:
        report["done"] = [public(j) for j in jobs]
        report["finished"] = True
        save_report()
        return report

    for j in [j for j in jobs if not j["sent"]]:  # nothing to score: empty unit files, no GPU
        j["written"] = bf.split_output(None, j["_preps"], out_root)
        report["done"].append(public(j))
    save_report()

    # One spawned call per group; each is split into its units as soon as it finishes, so a
    # run that stops early keeps every group already done. A rerun skips finished units, and
    # verify_biographies skips a group whose output exists, so a group scored but not yet
    # split is only re-split.
    import time

    rel = lambda p: os.path.relpath(p, DATA_DIR)  # noqa: E731
    # A group scored by an earlier run (e.g. one whose splitter stopped) is split here,
    # on CPU, rather than spawning a GPU call only to be told it is done.
    for j in [j for j in jobs if j["sent"] and os.path.exists(j["_out"])]:
        j["written"] = bf.split_output(j["_out"], j["_preps"], out_root)
        j["seconds"] = "scored earlier"
        report["done"].append(public(j))
    save_report()
    waiting = [(j, verify_biographies.spawn(rel(j["_inp"]), rel(j["_out"])))
               for j in jobs if j["sent"] and not os.path.exists(j["_out"])]
    while waiting:
        still = []
        for j, call in waiting:
            try:
                res = call.get(timeout=0)
            except Exception as e:  # noqa: BLE001  not finished yet, or the call failed
                if "Timeout" in type(e).__name__:
                    still.append((j, call))
                    continue
                j["error"] = "%s: %s" % (type(e).__name__, e)
                report["done"].append(public(j))
                save_report()
                continue
            data_vol.reload()
            j["written"] = bf.split_output(j["_out"], j["_preps"], out_root)
            j["seconds"] = res.get("seconds")
            report["done"].append(public(j))
            save_report()
        waiting = still
        if waiting:
            time.sleep(30)
    report["finished"] = True
    save_report()
    return report


@app.local_entrypoint()
def bio_full(gen: str = "generations", per_call: int = 4, max_groups: int = 0, dry_run: bool = False):
    """modal run --detach modal_app.py::bio_full [--max-groups 1] [--dry-run]

    The full Biographies verification, about $120 on H100 for all generations; run
    --dry-run first (CPU only), then --max-groups 1 --per-call 1.
    Writes results/bio_full_{gen}.json.
    """
    import json
    import pathlib

    report = bio_full_run.remote(gen, per_call, max_groups, dry_run)
    out = pathlib.Path(__file__).parent / f"results/bio_full_{gen}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    errors = [d for d in report["done"] if "error" in d]
    print("groups %d, responses %d, done %d, errors %d%s" % (
        report["groups"], report["responses"], len(report["done"]) - len(errors), len(errors),
        " (dry run)" if report["dry_run"] else ""))
    for d in errors:
        print("  ERROR", d["group"], d["error"])
    print("wrote", out)


@app.function(image=verifier_run_image, volumes={DATA_DIR: data_vol}, timeout=6 * 60 * 60, memory=8192)
def bio_redo_run(gen: str = "generations", per_call: int = 8, dry_run: bool = False) -> dict:
    """Verify again only the Biographies responses a tail-rule change touched (hallutrace/bio_redo.py).

    Plans once (redo_option_a_35/plan.json), so a rerun after a stop keeps the same groups and
    only splits groups already scored. After every group is scored, each changed unit's old
    files go to superseded/option_a_35/ and its new input, sidecar and merged output take
    their place. dry_run writes the plan and group inputs without calling Gemma.
    """
    import csv
    import inspect
    import json
    import os
    import time

    from hallutrace import verifier_inputs as bx
    from hallutrace import bio_full as bf
    from hallutrace import bio_redo as br
    from hallutrace.abstention import AbstentionDetector

    data_vol.reload()
    gen_root, out_root = f"{DATA_DIR}/{gen}", f"{DATA_DIR}/verifier/{gen}"
    redo_root = f"{out_root}/{br.REDO_DIR}"
    prompts = bx.load_prompts(f"{IMAGE_DATA}/prompts.csv")
    detector = AbstentionDetector("fixed")
    kw = {"send_abstained": False} if "send_abstained" in inspect.signature(bx.prepare).parameters else {}
    units = {u.unit_id: u for u in bx.discover(gen_root, "") if u.task == bx.BIOGRAPHIES}
    plan_path = f"{redo_root}/plan.json"

    if os.path.exists(plan_path):
        plan = json.load(open(plan_path, encoding="utf-8"))
    else:
        plan = {"tag": br.TAG, "units": {}}
        for uid, u in sorted(units.items()):
            if not os.path.exists(bx.output_path(out_root, u)):
                continue
            gids = br.changed(bx.prepare(u, prompts, detector, **kw), br.read_csv(br.sidecar_path(out_root, u)))
            if gids:
                plan["units"][uid] = sorted(gids)
        os.makedirs(redo_root, exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=1)
        data_vol.commit()

    news = {uid: bx.prepare(units[uid], prompts, detector, **kw) for uid in plan["units"]}
    subs = {uid: br.subset(news[uid], set(g)) for uid, g in plan["units"].items()}
    groups = bf.group_units([units[uid] for uid in plan["units"]], per_call)
    jobs = []
    for g in groups:
        inp, out = bf.group_paths(redo_root, g)
        preps = [subs[u.unit_id] for u in g]
        jobs.append({"group": bf.group_id(g), "units": len(g), "sent": bf.write_group_input(preps, inp),
                     "_preps": preps, "_inp": inp, "_out": out})
    data_vol.commit()
    report = {"gen": gen, "units": len(plan["units"]),
              "responses": sum(len(g) for g in plan["units"].values()), "groups": len(jobs),
              "by_model": {}, "dry_run": dry_run, "done": [], "merged": 0}
    for uid, g in plan["units"].items():
        m = units[uid].model
        report["by_model"][m] = report["by_model"].get(m, 0) + len(g)
    if dry_run:
        return report

    rel = lambda p: os.path.relpath(p, DATA_DIR)  # noqa: E731
    public = lambda j: {k: v for k, v in j.items() if not k.startswith("_")}  # noqa: E731
    for j in [j for j in jobs if os.path.exists(j["_out"])]:
        bf.split_output(j["_out"], j["_preps"], redo_root)
        report["done"].append(dict(public(j), seconds="scored earlier"))
    waiting = [(j, verify_biographies.spawn(rel(j["_inp"]), rel(j["_out"])))
               for j in jobs if not os.path.exists(j["_out"])]
    while waiting:
        still = []
        for j, call in waiting:
            try:
                res = call.get(timeout=0)
            except Exception as e:  # noqa: BLE001  not finished yet, or the call failed
                if "Timeout" in type(e).__name__:
                    still.append((j, call))
                    continue
                report["done"].append(dict(public(j), error="%s: %s" % (type(e).__name__, e)))
                continue
            data_vol.reload()
            bf.split_output(j["_out"], j["_preps"], redo_root)
            report["done"].append(dict(public(j), seconds=res.get("seconds")))
        waiting = still
        if waiting:
            time.sleep(20)
    data_vol.commit()
    if any("error" in d for d in report["done"]):
        return report  # merge nothing until every group is scored; a rerun finishes the rest

    for uid in plan["units"]:
        u = units[uid]
        kept = os.path.join(out_root, "superseded", br.TAG, "outputs", u.model, uid + ".csv")
        if os.path.exists(kept):
            continue  # merged by an earlier run; supersede would overwrite the original
        old_path = bx.output_path(out_root, u)
        old_rows = br.read_csv(old_path)
        new_rows = br.read_csv(bx.output_path(redo_root, u))
        merged = br.merge(old_rows, new_rows, news[uid], set(plan["units"][uid]))
        with open(old_path, encoding="utf-8", newline="") as f:
            columns = next(csv.reader(f), None) or list((new_rows or old_rows)[0].keys())
        bx.supersede(out_root, [u], br.TAG)
        bx.write_inputs(news[uid], out_root)
        br.write_rows(old_path, merged, columns)
        report["merged"] += 1
    data_vol.commit()
    return report


@app.local_entrypoint()
def bio_redo(gen: str = "generations", per_call: int = 8, dry_run: bool = False):
    """modal run modal_app.py::bio_redo [--dry-run]  ->  results/bio_redo_{gen}.json

    Re-scores only the Biographies responses that stopped on their own and lost their last
    line (or were emptied) under the old rule; 663 responses, about $10.
    Keep this process alive until it prints; a rerun resumes. Then run bio_filter again."""
    import json
    import pathlib

    report = bio_redo_run.remote(gen, per_call, dry_run)
    out = pathlib.Path(__file__).parent / f"results/bio_redo_{gen}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    errors = [d for d in report["done"] if "error" in d]
    print("units %d, responses %d, groups %d, done %d, errors %d, merged %d%s" % (
        report["units"], report["responses"], report["groups"], len(report["done"]) - len(errors),
        len(errors), report["merged"], " (dry run)" if report["dry_run"] else ""))
    print("by model", report["by_model"])
    for d in errors:
        print("  ERROR", d["group"], d["error"])
    print("wrote", out)


@app.function(image=verifier_run_image, volumes={DATA_DIR: data_vol}, timeout=2 * 60 * 60, memory=16384)
def bio_filter_run(gen: str = "generations") -> dict:
    """Write outputs_filtered/{model}/{unit}.csv next to every Biographies unit output:
    the same columns, with the atoms Gemma's generator invented removed (hallutrace/
    bio_filter.py), plus a `dropped atoms` column. The raw outputs are not touched.
    CPU only. The invented atoms are read from every FActScore cache on the volume at
    once: a reused container can write several groups' caches into one group's directory,
    so a unit's cache is not always under its own name."""
    import csv
    import glob
    import json
    import os
    import pickle

    from hallutrace import verifier_inputs as bx
    from hallutrace import bio_filter as bfl

    data_vol.reload()
    out_root = f"{DATA_DIR}/verifier/{gen}"
    inv, ref, kept_all = set(), set(), set()
    caches = sorted(glob.glob(f"{DATA_DIR}/factscore_cache/*/InstructGPT.pkl"))
    for pkl in caches:  # one at a time: each is a few hundred MB of score arrays
        with open(pkl, "rb") as f:
            i_, r_, k_ = bfl.invented_from_cache(pickle.load(f).values())
        inv |= i_
        ref |= r_
        kept_all |= k_
    inv -= kept_all  # an atom that is a real fact anywhere is never dropped
    totals = {"caches": len(caches), "units": 0, "responses": 0, "atoms": 0, "kept": 0,
              "false_raw": 0, "false_kept": 0, "header": 0, "invented": 0, "refusal": 0,
              "by_model": {}}
    for path in sorted(glob.glob(f"{out_root}/outputs/*/*.csv")):
        unit = os.path.splitext(os.path.basename(path))[0]
        model = os.path.basename(os.path.dirname(path))
        meta = f"{out_root}/inputs/{model}/{unit}.meta.csv"  # the sidecar names the task
        if not os.path.exists(meta) or next(csv.DictReader(open(meta, encoding="utf-8")), {}).get("task") != bx.BIOGRAPHIES:
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.getsize(path) else []
        out_rows = []
        for r in rows:
            atoms = bfl.parse_atoms(r["atoms"])
            kept, dropped = bfl.filter_atoms(atoms, inv, ref)
            totals["responses"] += 1
            totals["atoms"] += len(atoms)
            totals["kept"] += len(kept)
            totals["false_raw"] += sum(not v for _, v in atoms)
            totals["false_kept"] += sum(not v for _, v in kept)
            bm = totals["by_model"].setdefault(model, {"responses": 0, "atoms": 0, "kept": 0,
                                                        "false_raw": 0, "false_kept": 0})
            bm["responses"] += 1
            bm["atoms"] += len(atoms)
            bm["kept"] += len(kept)
            bm["false_raw"] += sum(not v for _, v in atoms)
            bm["false_kept"] += sum(not v for _, v in kept)
            for k in ("header", "invented", "refusal"):
                totals[k] += len(dropped[k])
            out_rows.append({**r, "atoms": bfl.format_atoms(kept),
                             "hallucinated atoms": bfl.format_atoms(kept, only_false=True),
                             "dropped atoms": json.dumps(dropped, ensure_ascii=False)})
        dst = f"{out_root}/outputs_filtered/{model}/{unit}.csv"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst + ".tmp", "w", encoding="utf-8", newline="") as f:
            if out_rows:
                w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
                w.writeheader()
                w.writerows(out_rows)
        os.replace(dst + ".tmp", dst)
        totals["units"] += 1
    data_vol.commit()
    return totals


@app.local_entrypoint()
def bio_filter(gen: str = "generations"):
    """modal run modal_app.py::bio_filter  ->  results/bio_filter_{gen}.json  (CPU, cents)"""
    import json
    import pathlib

    totals = bio_filter_run.remote(gen)
    out = pathlib.Path(__file__).parent / f"results/bio_filter_{gen}.json"
    out.write_text(json.dumps(totals, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(totals, indent=2))


# ---------------------------------------------------------------------------
# hand decomposition packets (checking the decomposer)
# ---------------------------------------------------------------------------

@app.function(image=verifier_run_image, volumes={DATA_DIR: data_vol}, timeout=30 * 60, memory=8192)
def decomp_cut_run(task: str, seed: str, gen: str = "generations", n: int = 30) -> dict:
    """Draw n generations of `task` for hand decomposition (hallutrace/decomp_cut.py).

    The text is what the verifier gets. CPU only; reads the volume and writes nothing to
    it. Returns the packet files' text."""
    from hallutrace import verifier_inputs as bx
    from hallutrace import decomp_cut as dc
    from hallutrace.abstention import AbstentionDetector

    data_vol.reload()
    prompts = bx.load_prompts(f"{IMAGE_DATA}/prompts.csv")
    gens = dc.candidates(f"{DATA_DIR}/{gen}", prompts, task, AbstentionDetector("fixed"))
    return dc.packet_files(dc.draw(gens, seed, task, n), seed, task, len(gens))


@app.local_entrypoint()
def decomp_cut(task: str, seed: str, gen: str = "generations", n: int = 30):
    """modal run modal_app.py::decomp_cut --task <task> --seed <seed>  ->  results/decomposition/<task>/

    CPU only. Packets stay local until they are sent to the annotator."""
    import json
    import pathlib

    files = decomp_cut_run.remote(task, seed, gen, n)
    d = pathlib.Path(__file__).parent / "results" / "decomposition" / task
    d.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8", newline="")
    m = json.loads(files["manifest.json"])
    print(json.dumps({k: v for k, v in m.items() if k != "instruction"}, indent=2, ensure_ascii=False))
    print("wrote", d)


# ---------------------------------------------------------------------------
# Biographies gold cut: the sample and the annotator packets
# ---------------------------------------------------------------------------

@app.function(image=verifier_run_image, volumes={DATA_DIR: data_vol, WIKI_DIR: wiki_vol},
              timeout=30 * 60, memory=8192)
def bio_gold_cut_run(seed: str, gen: str = "generations", n: int = 100) -> dict:
    """Draw the Biographies annotation sample from the filtered output and build the packets.

    Reads `/wiki/factscore/enwiki-20230401.db` read-only (sqlite mode=ro); writes
    nothing to either volume. Returns each packet file's text for the local side."""
    import os
    import tempfile

    from hallutrace import gold_cut as gc
    from hallutrace import gold_cut_bio as gb

    data_vol.reload()
    root = f"{DATA_DIR}/verifier/{gen}"
    units = gb.build_units(f"{root}/outputs_filtered", f"{root}/inputs")
    d = gc.draw(gb.TASK, units, seed, n=n)
    found, missing = gb.attach_articles(d.units, f"{FACTSCORE_DIR}/enwiki-20230401.db")
    d.stats.update({"articles_found": found, "articles_missing": missing,
                    "drawn_from": "outputs_filtered"})
    out = tempfile.mkdtemp()
    paths = gc.write_packets(d, out, sources=[u.source_file for u in d.units])
    return {name: open(p, encoding="utf-8").read() for name, p in paths.items()}


@app.local_entrypoint()
def bio_gold_cut(seed: str, gen: str = "generations", n: int = 100):
    """modal run modal_app.py::bio_gold_cut --seed <seed>  ->  results/gold_cut/biographies/

    CPU only. Record the seed with the draw."""
    import json
    import pathlib

    files = bio_gold_cut_run.remote(seed, gen, n)
    d = pathlib.Path(__file__).parent / "results" / "gold_cut" / "biographies"
    d.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8", newline="")
    print(json.dumps(json.loads(files["manifest.json"])["stats"], indent=2, ensure_ascii=False))
    print("wrote", d)


@app.function(image=verifier_run_image, volumes={DATA_DIR: data_vol}, timeout=30 * 60, memory=16384)
def bio_passages_run(items: list) -> dict:
    """The 5 retrieved passages per gold-cut atom, from every FActScore retrieval cache.

    hallutrace/bio_passages.py. CPU only; reads the volume, writes nothing to it."""
    import glob
    import json

    from hallutrace import bio_passages as bp

    data_vol.reload()
    paths = sorted(glob.glob(f"{BIO_CACHE_ROOT}/*/retrieval-enwiki-20230401.json"))

    def caches():
        for path in paths:
            with open(path, encoding="utf-8") as f:
                yield json.load(f)

    found = bp.lookup(caches(), items)
    return {"caches": len(paths), "passages": {k: bp.clean(v) for k, v in found.items()}}


@app.local_entrypoint()
def bio_passages():
    """modal run modal_app.py::bio_passages  ->  results/gold_cut/biographies/passages.json

    For the Biographies annotation packet already drawn: what the validator saw for each atom, so
    kappa can be split into retrieval and judgement. A key-side file: never send it
    to annotators. CPU, cents."""
    import csv
    import json
    import pathlib

    d = pathlib.Path(__file__).parent / "results" / "gold_cut" / "biographies"
    with open(d / "annotator_A.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    items = [{"item_id": r["item_id"], "topic": r["context_shown"].split("\n", 1)[0],
              "atom": r["unit_to_judge"]} for r in rows]
    res = bio_passages_run.remote(items)
    out = {it["item_id"]: {"topic": it["topic"], "atom": it["atom"],
                           "passages": res["passages"][it["item_id"]]} for it in items}
    (d / "passages.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    missing = [k for k, v in out.items() if v["passages"] is None]
    print("caches read %d; passages found for %d of %d atoms" % (res["caches"], len(out) - len(missing), len(out)))
    if missing:
        print("not found:", missing[:10])
    print("wrote", d / "passages.json")
