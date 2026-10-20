"""The pinned model revisions agree everywhere they are written.

    python -m tests.test_model_revisions

No network, no Modal call: imports modal_app for its constants only.
"""

import glob
import json
import pathlib
import re

from modal_app import MODEL_REVISIONS, SUBJECT_MODELS, _with_revision

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    checks = 0
    for name, rev in MODEL_REVISIONS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", rev), f"{name}: not a full commit hash: {rev}"
        checks += 1
    assert set(SUBJECT_MODELS) <= set(MODEL_REVISIONS), "every subject model must be pinned"
    checks += 1

    manifest = json.loads((ROOT / "data/generation_plan/plan_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_revisions"] == {m: MODEL_REVISIONS[m] for m in SUBJECT_MODELS}, \
        "plan_manifest.json model_revisions drifted from modal_app.MODEL_REVISIONS"
    checks += 1

    for path in glob.glob(str(ROOT / "results/pilot_1*/*/*.jsonl")):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            assert rec.get("model_revision") == MODEL_REVISIONS[rec["model"]], \
                f"{path}: {rec['generation_id']} has model_revision {rec.get('model_revision')}"
            checks += 1

    rec = {"generation_id": "x", "model": "olmo2-7b", "engine": "e", "gpu": "g", "n_prompt_tokens": 1}
    out = _with_revision(rec)
    assert list(out) == ["generation_id", "model", "engine", "gpu", "model_revision", "n_prompt_tokens"]
    assert _with_revision(out) == out, "backfill must be idempotent"
    try:
        _with_revision(out | {"model_revision": "0" * 40})
        raise AssertionError("a wrong revision must be refused")
    except ValueError:
        pass
    checks += 3

    print(f"all {checks} checks passed")


if __name__ == "__main__":
    main()
