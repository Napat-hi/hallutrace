"""One call signature for all of HALoGEN's verifiers.

Replaces HALoGEN's src/evaluate_hallucinations.py, which cannot run as released:
four branches pass model_name= to classes that take model=, the input path is
fixed to one model's file, and the verifiers write their output to different
places (some into the parent directory or the working directory). This wrapper
calls the same classes and methods, so verification logic is unchanged, and
puts each output at the path asked for.

Biographies raises on a repeated name only after scoring every fact, so the
wrapper refuses such a frame first, and by default runs the verifier once per
(model, sample_index).

Pass `client=` (an llm_client.LLMClient) to route every LLM call through it.

    run_verifier(task="historical_events",
                 input_csv="run/olmo2_7b_generations_historical_events.csv",
                 output_csv="run/olmo2_7b_units_historical_events.csv")

Keys come from the environment: TOGETHER_API_KEY (not needed with a client)
and S2_API_KEY.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd

from hallutrace import llm_client
from hallutrace import cache
from hallutrace.semantic_scholar import SemanticScholarClient

DEFAULT_VERIFIER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def _env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        raise RuntimeError(
            "%s is not set. Export it before running; do not write it into "
            "config.yml, the repo is public." % name
        )
    return v


@dataclass
class TaskSpec:
    """One HALoGEN task, and everything that differs about it."""

    name: str
    module: str
    cls: str
    required_columns: Tuple[str, ...]
    keys: Tuple[str, ...]
    # the verifier's output file name, given the input stem
    out_name: Callable[[str], str]
    # "argdir": the method takes output_dir; "cwd": it writes into the CWD
    out_where: str
    ctor_kind: str  # "none" | "together" | "decomp" | "s2"
    method: str = "evaluate"
    notes: str = ""
    # columns that must be unique within one call (empty: no constraint)
    unique_by: Tuple[str, ...] = ()
    # columns to split the input on so unique_by holds (empty: never split)
    shard_by: Tuple[str, ...] = ()


def _au(stem: str) -> str:
    return stem + "_AU.csv"


def _same(stem: str) -> str:
    return stem + ".csv"


TASKS = {
    "biographies": TaskSpec(
        name="biographies",
        module="verifiers.biographies",
        cls="BiographiesEvaluator",
        required_columns=("name", "response"),
        keys=(),
        out_name=_same,
        out_where="argdir",
        ctor_kind="none",
        method="evaluate",
        unique_by=("name",),
        shard_by=("model", "sample_index"),
        notes=(
            "Wraps OpenFActScore after patch halogen-openfactscore. No "
            "API key: both the atomic fact generator and the validator are "
            "local Hugging Face models, set through HALLUTRACE_AFG_MODEL and "
            "HALLUTRACE_AFV_MODEL. Needs the enwiki knowledge source at "
            "HALLUTRACE_FACTSCORE_DIR. Output keeps the input filename, so the "
            "input and output directories must differ. The name "
            "must be unique within one call, so the wrapper shards on "
            "(model, sample_index) and refuses a frame that still repeats."
        ),
    ),
    "code_packages": TaskSpec(
        name="code_packages",
        module="verifiers.code_packages",
        cls="CodeEvaluator",
        required_columns=("response",),
        keys=(),
        out_name=_au,
        out_where="argdir",
        ctor_kind="none",
        notes="No LLM. Queries PyPI for each imported package name.",
    ),
    "false_presuppositions": TaskSpec(
        name="false_presuppositions",
        module="verifiers.false_presuppositions",
        cls="NumericalFalsePresuppostionEvaluator",
        required_columns=("response", "list", "list_condition"),
        keys=("TOGETHER_API_KEY",),
        out_name=_au,
        out_where="argdir",
        ctor_kind="together",
        notes=(
            "list and list_condition are read with ast.literal_eval, so they "
            "must be written as Python list literals, not JSON."
        ),
    ),
    "historical_events": TaskSpec(
        name="historical_events",
        module="verifiers.historical_events",
        cls="InterpersonalRelationshipEvaluator",
        required_columns=("response", "name1", "name2"),
        keys=("TOGETHER_API_KEY",),
        out_name=_au,
        out_where="argdir",
        ctor_kind="together",
        notes=(
            "Pre filters on the literal substring 'meeting' before asking the "
            "verifier anything (historical_events.py line 53). A response that "
            "describes a meeting without using the word is scored as a non "
            "response. The prompt at line 56 misspells confirm as comfirm."
        ),
    ),
    "rationalization_binary": TaskSpec(
        name="rationalization_binary",
        module="verifiers.rationalization_binary",
        cls="YesNoResponseEvaluator",
        required_columns=("response", "category", "prompt"),
        keys=("TOGETHER_API_KEY",),
        out_name=_au,
        out_where="argdir",
        ctor_kind="together",
        notes=(
            "category must be one of primality, graph, senator. Any other "
            "value leaves the extraction prompt None and the row is skipped. "
            "Needs spacy en_core_web_sm at construction time."
        ),
    ),
    "rationalization_numerical": TaskSpec(
        name="rationalization_numerical",
        module="verifiers.rationalization_numerical",
        cls="NumAutoregressiveEvaluator",
        required_columns=("response", "list", "list_condition", "count"),
        keys=("TOGETHER_API_KEY",),
        out_name=_au,
        out_where="argdir",
        ctor_kind="together",
    ),
    "simplification": TaskSpec(
        name="simplification",
        module="verifiers.simplification",
        cls="SimplificationEvaluator",
        required_columns=(),
        keys=("TOGETHER_API_KEY",),
        out_name=_au,
        out_where="cwd",
        ctor_kind="decomp",
        notes=(
            "Reads by POSITION, not by column name (simplification.py lines "
            "158 to 163). Four columns means passage=col 1, response=col 3. "
            "Any other width means passage=col 0, response=col 1. The wrapper "
            "enforces the two column shape. evaluate() takes no output_dir at "
            "all and writes into the process CWD."
        ),
    ),
    "summarization": TaskSpec(
        name="summarization",
        module="verifiers.summarization",
        cls="SummarizationEvaluator",
        required_columns=(),
        keys=("TOGETHER_API_KEY",),
        out_name=_au,
        out_where="cwd",
        ctor_kind="decomp",
        notes=(
            "Same positional reading as simplification (lines 148 to 153). "
            "evaluate() accepts output_dir and then ignores it (line 189). "
            "Prints every row to stdout."
        ),
    ),
    "scientific_attribution": TaskSpec(
        name="scientific_attribution",
        module="scientific_attribution",
        cls="ReferencesEvaluator",
        required_columns=("response",),
        keys=("S2_API_KEY", "TOGETHER_API_KEY"),
        out_name=_same,
        out_where="argdir",
        ctor_kind="s2",
        method="evaluate_references",
        notes=(
            "Lives in src/, not verifiers/, and the released entrypoint has "
            "pass in this branch. Method is evaluate_references, not evaluate. "
            "Output keeps the input filename. Holds the only correct copy of "
            "the abstention phrase list; see abstention.py."
        ),
    ),
}

# the two columns summarization and simplification read by position
_POSITIONAL_COLUMNS = ("prompt", "response")


# _query_semantic_scholar is the only HALoGEN method that calls Semantic Scholar.
# _build_s2 overrides it in a subclass to use SemanticScholarClient (paced, with
# backoff) and the persistent cache, so a resumed run repeats no lookups.
def _build_s2(klass, model: str, transport=None, together_api_key=None):
    """Build klass (ReferencesEvaluator) with the S2 client and the cache.

    The cache path is made absolute and opened now, because _run_once later
    changes into a temporary directory and deletes it. _run_once closes the
    connection. `transport` is for tests.
    """
    cache_path = os.path.abspath(os.environ.get("HALLUTRACE_CACHE_PATH", cache.DEFAULT_CACHE_PATH))
    resilient = cache.ResilientCache(cache_path, cache.SemanticScholarCache)

    class _S2Wired(klass):
        def _query_semantic_scholar(self, title):
            return self._hallutrace_s2_client.title_for(title)

    evaluator = _S2Wired(
        s2_api_key=_env("S2_API_KEY"),
        together_api_key=together_api_key or _env("TOGETHER_API_KEY"),
        model=model,
    )
    evaluator._hallutrace_s2_client = SemanticScholarClient(
        api_key=_env("S2_API_KEY"), cache=resilient, transport=transport,
    )
    evaluator._hallutrace_s2_cache = resilient  # closed by _run_once's finally
    return evaluator


def _build(spec: TaskSpec, model: str, keydir: str, client=None):
    mod = __import__(spec.module, fromlist=[spec.cls])
    klass = getattr(mod, spec.cls)

    def together_key() -> str:
        # with a client, Together(...) is routed to it and needs no key
        return "routed-to-client" if client is not None else _env("TOGETHER_API_KEY")

    if spec.ctor_kind == "none":
        return klass()
    if spec.ctor_kind == "together":
        # model=, not the released entrypoint's model_name=
        return klass(api_key=together_key(), model=model)
    if spec.ctor_kind == "decomp":
        if "gpt" in model:
            # get_response_target sends any name containing "gpt" to OpenAI.
            raise ValueError(
                "%s would send %r to OpenAI; this project uses no OpenAI "
                "model. Use a Together hosted model." % (spec.name, model)
            )
        # one model for both stages
        return klass(
            decomposition_api_key=together_key(),
            entailment_api_key=together_key(),
            decomposition_model=model,
            entailment_model=model,
        )
    if spec.ctor_kind == "s2":
        if client is None:
            return _build_s2(klass, model)
        # Together(...) goes to the client, so the key is a placeholder
        return _build_s2(klass, model, together_api_key=together_key())
    raise ValueError("unknown ctor_kind %r" % spec.ctor_kind)


def _check_columns(spec: TaskSpec, df: pd.DataFrame) -> None:
    missing = [c for c in spec.required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "%s needs column(s) %s; the input has %s"
            % (spec.name, missing, list(df.columns))
        )
    if spec.out_where == "cwd":
        # positional readers
        if list(df.columns) != list(_POSITIONAL_COLUMNS):
            raise ValueError(
                "%s reads its input by position. Give it exactly the columns "
                "%s in that order; got %s"
                % (spec.name, list(_POSITIONAL_COLUMNS), list(df.columns))
            )


def _check_unique(spec: TaskSpec, df: pd.DataFrame, where: str) -> None:
    """Refuse a frame whose unique_by key repeats, before any work is done.

    Biographies raises on a repeated name only after scoring every atomic fact.
    The message names the most repeated values and their counts.
    """
    if not spec.unique_by:
        return
    missing = [c for c in spec.unique_by if c not in df.columns]
    if missing:
        # _check_columns enforces required columns; nothing to check here
        return

    counts = df.groupby(list(spec.unique_by)).size()
    repeated = counts[counts > 1]
    if repeated.empty:
        return

    worst = repeated.sort_values(ascending=False).head(5)
    examples = ", ".join("%r x%d" % (k, int(v)) for k, v in worst.items())
    hint = ""
    if spec.shard_by:
        have = [c for c in spec.shard_by if c in df.columns]
        if have:
            hint = (
                " The input has %s, so pass shard=True (the default) and the "
                "wrapper will split on it for you." % ", ".join(have)
            )
        else:
            hint = (
                " %s carries none of the sharding columns %s, so the wrapper "
                "cannot split it. Split the file so that each call holds one "
                "(model, sample_index)." % (where, ", ".join(spec.shard_by))
            )

    raise ValueError(
        "%s requires %s to be unique within one call, and %s repeats %d of "
        "them (%d rows in total). Worst: %s. HALoGEN would raise "
        "InvalidIndexError on this *after* scoring every atomic fact, "
        "so it is refused here instead.%s"
        % (
            spec.name,
            " + ".join(spec.unique_by),
            where,
            len(repeated),
            int(repeated.sum()),
            examples,
            hint,
        )
    )


def _shard_label(values: Tuple) -> str:
    """A filesystem-safe label for one shard, for temporary file names."""
    parts = []
    for v in values:
        s = str(v)
        parts.append("".join(ch if ch.isalnum() else "_" for ch in s))
    return "-".join(parts)


def _split(spec: TaskSpec, df: pd.DataFrame, shard: bool) -> List[Tuple[str, dict, pd.DataFrame]]:
    """The pieces to run, as (label, key values, frame).

    A single piece with an empty key means the whole frame.
    """
    if not shard or not spec.shard_by:
        return [("", {}, df)]
    have = [c for c in spec.shard_by if c in df.columns]
    if len(have) != len(spec.shard_by):
        return [("", {}, df)]

    pieces = []
    # sorted, so the concatenated output is the same on every run
    for values, frame in sorted(df.groupby(list(spec.shard_by), sort=True)):
        if not isinstance(values, tuple):
            values = (values,)
        pieces.append((_shard_label(values), dict(zip(spec.shard_by, values)), frame))
    return pieces


def shard_plan(task: str, input_csv: str, shard: bool = True) -> List[dict]:
    """The shards run_verifier would run, with their row counts. No network."""
    if task not in TASKS:
        raise KeyError("unknown task %r; known: %s" % (task, sorted(TASKS)))
    spec = TASKS[task]
    df = pd.read_csv(input_csv)
    out = []
    for label, keys, frame in _split(spec, df, shard):
        out.append({"label": label, "keys": keys, "rows": int(len(frame))})
    return out


def _run_once(
    spec: TaskSpec,
    df: pd.DataFrame,
    stem: str,
    out_abs: str,
    model: str,
    client=None,
) -> None:
    """Run the verifier on one frame and move its output to out_abs."""
    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="hallutrace_%s_" % spec.name)
    undo = None
    evaluator = None
    try:
        if client is not None:
            mod = __import__(spec.module, fromlist=[spec.cls])
            undo = llm_client.route([mod], client)
        staged_in = os.path.join(tmp, stem + ".csv")
        df.to_csv(staged_in, index=False)
        outdir = os.path.join(tmp, "out")
        os.makedirs(outdir)

        evaluator = _build(spec, model, tmp, client)

        # some verifiers write into the working directory
        os.chdir(tmp)

        if spec.out_where == "cwd":
            evaluator.evaluate(staged_in)
        else:
            getattr(evaluator, spec.method)(staged_in, outdir + os.sep)

        wanted = spec.out_name(stem)
        produced = None
        for cand in (
            os.path.join(outdir, wanted),
            os.path.join(tmp, wanted),
            os.path.join(os.path.dirname(outdir), wanted),
        ):
            if os.path.isfile(cand):
                produced = cand
                break
        if produced is None:
            listing = []
            for d, _, fs in os.walk(tmp):
                listing.extend(os.path.join(d, f) for f in fs)
            raise RuntimeError(
                "%s wrote no file named %s. Files under the temp dir: %s"
                % (spec.name, wanted, listing)
            )
        shutil.move(produced, out_abs)
    finally:
        if undo is not None:
            undo()
        s2_cache = getattr(evaluator, "_hallutrace_s2_cache", None)
        if s2_cache is not None:
            if s2_cache.failures:
                status = (
                    "disabled, no cache for the rest of this run" if s2_cache.disabled
                    else "reconnect budget spent (%d), retrying without reconnecting"
                         % s2_cache.MAX_RECONNECTS if s2_cache.reconnect_budget_exhausted
                    else "recovered"
                )
                print(
                    "S2 cache: %d failure(s) this run, %d reconnect(s), %s"
                    % (s2_cache.failures, s2_cache.reconnects, status)
                )
            s2_cache.close()
        os.chdir(prev_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def run_verifier(
    task: str,
    input_csv: str,
    output_csv: str,
    model: str = DEFAULT_VERIFIER_MODEL,
    repo_root: Optional[str] = None,
    dry_run: bool = False,
    shard: bool = True,
    client=None,
) -> str:
    """Run one HALoGEN verifier and write its output to output_csv.

    Returns output_csv; raises if the verifier produced nothing. With
    `shard=True` a task whose spec has shard_by (Biographies) runs once per
    piece and the results are concatenated. `client` (an llm_client.LLMClient)
    takes every LLM call the verifier makes.
    """
    if task not in TASKS:
        raise KeyError("unknown task %r; known: %s" % (task, sorted(TASKS)))
    spec = TASKS[task]

    root = repo_root or os.environ.get("HALOGEN_ROOT", "")
    if not root or not os.path.isdir(root):
        raise RuntimeError(
            "Set HALOGEN_ROOT to the cloned HALoGEN checkout, or pass repo_root."
        )
    # summarization and simplification import get_model_responses from verifiers/
    for sub in ("", "src", "verifiers"):
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)

    df = pd.read_csv(input_csv)
    _check_columns(spec, df)
    for k in spec.keys:
        if client is not None and k == "TOGETHER_API_KEY":
            continue
        _env(k)

    pieces = _split(spec, df, shard)

    # checked per piece, and also on a dry run
    for label, keys, frame in pieces:
        where = "the input" if not label else "shard %s of the input" % label
        _check_unique(spec, frame, where)

    if dry_run:
        return output_csv

    out_abs = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)

    stem = os.path.splitext(os.path.basename(input_csv))[0]

    if len(pieces) == 1:
        _run_once(spec, pieces[0][2], stem, out_abs, model, client)
        return output_csv

    # run each piece into its own file, then concatenate with the shard keys
    holding = tempfile.mkdtemp(prefix="hallutrace_shards_%s_" % task)
    try:
        frames = []
        for label, keys, frame in pieces:
            piece_out = os.path.join(holding, "%s__%s.csv" % (stem, label))
            _run_once(spec, frame, "%s__%s" % (stem, label), piece_out, model, client)
            got = pd.read_csv(piece_out)
            for col, value in keys.items():
                if col not in got.columns:
                    got[col] = value
            frames.append(got)
        pd.concat(frames, ignore_index=True, sort=False).to_csv(out_abs, index=False)
    finally:
        shutil.rmtree(holding, ignore_errors=True)

    return output_csv


def describe(task: Optional[str] = None) -> str:
    """What a task needs, as text. No imports, no network."""
    names = [task] if task else sorted(TASKS)
    out: List[str] = []
    for n in names:
        s = TASKS[n]
        out.append(
            "%s\n  class    %s.%s.%s()\n  columns  %s\n  keys     %s\n"
            "  output   %s, written %s\n  unique   %s\n  shard    %s\n"
            "  note     %s"
            % (
                s.name,
                s.module,
                s.cls,
                s.method,
                ", ".join(s.required_columns) or "(positional, see note)",
                ", ".join(s.keys) or "(none)",
                s.out_name("<stem>"),
                "to the directory argument" if s.out_where == "argdir" else "into the CWD",
                " + ".join(s.unique_by) or "(no constraint)",
                " + ".join(s.shard_by) or "(never split)",
                s.notes or "(none)",
            )
        )
    return "\n\n".join(out)


if __name__ == "__main__":
    print(describe(sys.argv[1] if len(sys.argv) > 1 else None))
