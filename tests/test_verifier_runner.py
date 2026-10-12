"""Tests for verifier_runner.py. No network, no API keys, no Together import."""

import os
import shutil
import tempfile

import pandas as pd

from hallutrace import verifier_runner as vr

OK = "   OK  "
FAIL = "   FAIL"
failures = []


def check(cond, label):
    print((OK if cond else FAIL) + " " + label)
    if not cond:
        failures.append(label)


def expect_raises(exc, fn, label, substr=None):
    try:
        fn()
    except exc as e:
        if substr and substr not in str(e):
            check(False, label + "  (wrong message: %s)" % e)
        else:
            check(True, label)
        return
    except Exception as e:  # noqa: BLE001
        check(False, label + "  (raised %s instead: %s)" % (type(e).__name__, e))
        return
    check(False, label + "  (did not raise)")


print("1. registry")
check(len(vr.TASKS) == 9, "nine tasks registered, got %d" % len(vr.TASKS))
check(
    vr.TASKS["scientific_attribution"].method == "evaluate_references",
    "scientific_attribution uses evaluate_references",
)
check(
    vr.TASKS["biographies"].out_name("x") == "x.csv"
    and vr.TASKS["code_packages"].out_name("x") == "x_AU.csv",
    "the two output naming conventions are both represented",
)
check(
    {t for t, s in vr.TASKS.items() if s.out_where == "cwd"}
    == {"summarization", "simplification"},
    "exactly the two CWD writers are marked as such",
)
check(len(vr.describe()) > 500, "describe() renders every task")

print("2. argument validation")
expect_raises(KeyError, lambda: vr.run_verifier("nope", "a.csv", "b.csv"), "unknown task")

tmp = tempfile.mkdtemp(prefix="hallutrace_test_")
try:
    os.environ.setdefault("TOGETHER_API_KEY", "test-not-a-real-key")
    os.environ.setdefault("OPENAI_API_KEY", "test-not-a-real-key")
    os.environ.setdefault("S2_API_KEY", "test-not-a-real-key")
    root = os.path.join(tmp, "fakerepo", "src")
    os.makedirs(root)

    bad = os.path.join(tmp, "bad.csv")
    pd.DataFrame({"response": ["x"]}).to_csv(bad, index=False)
    expect_raises(
        ValueError,
        lambda: vr.run_verifier(
            "historical_events", bad, os.path.join(tmp, "o.csv"),
            repo_root=os.path.join(tmp, "fakerepo"), dry_run=True,
        ),
        "missing columns are named",
        substr="name1",
    )

    wide = os.path.join(tmp, "wide.csv")
    pd.DataFrame({"a": [1], "prompt": ["p"], "b": [2], "response": ["r"]}).to_csv(
        wide, index=False
    )
    expect_raises(
        ValueError,
        lambda: vr.run_verifier(
            "summarization", wide, os.path.join(tmp, "o.csv"),
            repo_root=os.path.join(tmp, "fakerepo"), dry_run=True,
        ),
        "positional readers reject a four column frame",
        substr="by position",
    )

    narrow = os.path.join(tmp, "narrow.csv")
    pd.DataFrame({"prompt": ["p"], "response": ["r"]}).to_csv(narrow, index=False)
    vr.run_verifier(
        "summarization", narrow, os.path.join(tmp, "o.csv"),
        repo_root=os.path.join(tmp, "fakerepo"), dry_run=True,
    )
    check(True, "positional readers accept the two column frame")

    expect_raises(
        RuntimeError,
        lambda: vr.run_verifier(
            "code_packages", narrow, os.path.join(tmp, "o.csv"), repo_root="/no/such/dir"
        ),
        "a bad repo_root is reported, not ignored",
        substr="HALOGEN_ROOT",
    )

    print("3. missing key is reported by name")
    saved = os.environ.pop("S2_API_KEY")
    sci = os.path.join(tmp, "sci.csv")
    pd.DataFrame({"response": ["r"]}).to_csv(sci, index=False)
    expect_raises(
        RuntimeError,
        lambda: vr.run_verifier(
            "scientific_attribution", sci, os.path.join(tmp, "o.csv"),
            repo_root=os.path.join(tmp, "fakerepo"), dry_run=True,
        ),
        "S2_API_KEY named in the error",
        substr="S2_API_KEY",
    )
    os.environ["S2_API_KEY"] = saved

    print("4. output lands exactly where we asked, whatever the verifier does")

    class _WritesToCwd:
        """Behaves like summarization: bare filename, into the CWD."""

        def evaluate(self, csv_path, output_dir=None):
            stem = os.path.basename(csv_path).replace(".csv", "")
            pd.DataFrame({"atoms": ["a | True"]}).to_csv(stem + "_AU.csv", index=False)

    class _WritesNothing:
        def evaluate(self, csv_path, output_dir=None):
            return None

    real_build = vr._build
    outer_cwd = os.getcwd()

    vr._build = lambda spec, model, keydir, client=None: _WritesToCwd()
    dest = os.path.join(tmp, "nested", "units_summarization.csv")
    got = vr.run_verifier(
        "summarization", narrow, dest, repo_root=os.path.join(tmp, "fakerepo")
    )
    check(got == dest and os.path.isfile(dest), "CWD writer output relocated to output_csv")
    check(
        list(pd.read_csv(dest).columns) == ["atoms"],
        "relocated file is the verifier's output, not a copy of the input",
    )
    check(os.getcwd() == outer_cwd, "cwd restored after a successful run")

    vr._build = lambda spec, model, keydir, client=None: _WritesNothing()
    expect_raises(
        RuntimeError,
        lambda: vr.run_verifier(
            "summarization", narrow, os.path.join(tmp, "never.csv"),
            repo_root=os.path.join(tmp, "fakerepo"),
        ),
        "a verifier that writes nothing fails loudly",
        substr="wrote no file",
    )
    check(os.getcwd() == outer_cwd, "cwd restored after a failed run")
    check(
        not os.path.isfile(os.path.join(tmp, "never.csv")),
        "no empty output file left behind on failure",
    )

    print("5. the input is never overwritten")
    before = open(narrow).read()

    class _OverwritesInput:
        """Behaves like biographies: output keeps the input filename."""

        def evaluate(self, csv_path, output_dir=None):
            stem = os.path.basename(csv_path).replace(".csv", "")
            pd.DataFrame({"atoms": ["a"]}).to_csv(
                os.path.join(output_dir, stem + "_AU.csv"), index=False
            )

    vr._build = lambda spec, model, keydir, client=None: _OverwritesInput()
    vr.run_verifier(
        "code_packages", narrow, os.path.join(tmp, "code_units.csv"),
        repo_root=os.path.join(tmp, "fakerepo"),
    )
    check(open(narrow).read() == before, "input csv unchanged after a run")

    print("6. biographies must never see a repeated name")

    calls = []

    class _Biographies:
        """Stands in for BiographiesEvaluator: writes <stem>.csv into output_dir.

        The final join indexes on name, as biographies.py does, so a repeated
        name raises after scoring, as on a real run. The exception type depends
        on the pandas version (InvalidIndexError or ValueError).
        """

        def evaluate(self, csv_path, output_dir=None):
            frame = pd.read_csv(csv_path)
            calls.append(list(frame["name"]))
            scored = pd.DataFrame(
                {"name": frame["name"].unique(), "score": 0.5}
            )
            scored["response"] = frame.set_index("name")["response"]
            stem = os.path.basename(csv_path).replace(".csv", "")
            scored.to_csv(os.path.join(output_dir, stem + ".csv"), index=False)

    vr._build = lambda spec, model, keydir, client=None: _Biographies()

    # two models x two sample_index, the same two people in each: names
    # repeat in the file but not inside a shard
    sharded = os.path.join(tmp, "bio_sharded.csv")
    rows = []
    for m in ("olmo2-7b", "pythia-69b"):
        for si in (0, 1):
            for who in ("Ada Lovelace", "Charles Babbage"):
                rows.append({"name": who, "response": "text", "model": m,
                             "sample_index": si})
    pd.DataFrame(rows).to_csv(sharded, index=False)

    plan = vr.shard_plan("biographies", sharded)
    check(len(plan) == 4, "shard_plan sees four shards, got %d" % len(plan))
    check(
        all(p["rows"] == 2 for p in plan),
        "every shard holds two rows, so the model load cost is paid four times",
    )
    check(
        plan[0]["keys"] == {"model": "olmo2-7b", "sample_index": 0},
        "shard keys are reported, so a run can be costed before it starts",
    )

    del calls[:]
    dest = os.path.join(tmp, "bio_units.csv")
    vr.run_verifier(
        "biographies", sharded, dest, repo_root=os.path.join(tmp, "fakerepo")
    )
    check(len(calls) == 4, "the verifier ran once per shard, got %d" % len(calls))
    check(
        all(len(set(c)) == len(c) for c in calls),
        "no shard ever carried a repeated name, which is the whole point",
    )
    out = pd.read_csv(dest)
    check(len(out) == 8, "every input row is in the concatenated output, got %d" % len(out))
    check(
        {"model", "sample_index"} <= set(out.columns),
        "the shard keys are put back on the output rows, so it joins to table 2",
    )
    check(
        sorted(out["model"].unique()) == ["olmo2-7b", "pythia-69b"]
        and sorted(out["sample_index"].unique()) == [0, 1],
        "both models and both sample_index values survive the concatenation",
    )

    print("   the guard fires before any scoring, not after")

    # A duplicate that sharding cannot remove: the same person twice inside one
    # (model, sample_index).
    dup_in_shard = os.path.join(tmp, "bio_dup_in_shard.csv")
    pd.DataFrame(
        {
            "name": ["Ada Lovelace", "Ada Lovelace"],
            "response": ["a", "b"],
            "model": ["olmo2-7b", "olmo2-7b"],
            "sample_index": [1, 1],
        }
    ).to_csv(dup_in_shard, index=False)
    del calls[:]
    expect_raises(
        ValueError,
        lambda: vr.run_verifier(
            "biographies", dup_in_shard, os.path.join(tmp, "never2.csv"),
            repo_root=os.path.join(tmp, "fakerepo"),
        ),
        "a name repeated inside one shard is refused",
        substr="Ada Lovelace",
    )
    check(calls == [], "the verifier was never built, so no GPU time was spent")
    check(
        not os.path.isfile(os.path.join(tmp, "never2.csv")),
        "and no partial output was written",
    )

    # no sharding columns and repeated names: the wrapper cannot split it
    unsplittable = os.path.join(tmp, "bio_flat.csv")
    pd.DataFrame(
        {"name": ["Ada Lovelace", "Ada Lovelace"], "response": ["a", "b"]}
    ).to_csv(unsplittable, index=False)
    expect_raises(
        ValueError,
        lambda: vr.run_verifier(
            "biographies", unsplittable, os.path.join(tmp, "never3.csv"),
            repo_root=os.path.join(tmp, "fakerepo"),
        ),
        "a frame with no sharding columns says how to split it",
        substr="one (model, sample_index)",
    )

    expect_raises(
        ValueError,
        lambda: vr.run_verifier(
            "biographies", sharded, os.path.join(tmp, "never4.csv"),
            repo_root=os.path.join(tmp, "fakerepo"), shard=False,
        ),
        "shard=False still guards, it does not disable the check",
        substr="DEVIATIONS 13",
    )

    expect_raises(
        ValueError,
        lambda: vr.run_verifier(
            "biographies", dup_in_shard, os.path.join(tmp, "never5.csv"),
            repo_root=os.path.join(tmp, "fakerepo"), dry_run=True,
        ),
        "dry_run guards too, which is the point of running it first",
        substr="DEVIATIONS 13",
    )

    print("   a single shard is not routed through the concatenation path")
    one_shard = os.path.join(tmp, "bio_one.csv")
    pd.DataFrame(
        {
            "name": ["Ada Lovelace", "Charles Babbage"],
            "response": ["a", "b"],
            "model": ["olmo2-7b", "olmo2-7b"],
            "sample_index": [1, 1],
        }
    ).to_csv(one_shard, index=False)
    del calls[:]
    dest1 = os.path.join(tmp, "bio_one_units.csv")
    vr.run_verifier(
        "biographies", one_shard, dest1, repo_root=os.path.join(tmp, "fakerepo")
    )
    check(len(calls) == 1, "one shard means one call, got %d" % len(calls))
    check(len(pd.read_csv(dest1)) == 2, "and the output is the verifier's, unchanged")

    print("7. every other task is untouched by any of this")
    check(
        [t for t, s in vr.TASKS.items() if s.shard_by] == ["biographies"],
        "biographies is the only task that shards",
    )
    check(
        [t for t, s in vr.TASKS.items() if s.unique_by] == ["biographies"],
        "and the only task with a uniqueness constraint",
    )

    vr._build = lambda spec, model, keydir, client=None: _OverwritesInput()
    repeated = os.path.join(tmp, "repeats.csv")
    pd.DataFrame({"prompt": ["p", "p"], "response": ["r", "r"]}).to_csv(
        repeated, index=False
    )
    vr.run_verifier(
        "code_packages", repeated, os.path.join(tmp, "code_units2.csv"),
        repo_root=os.path.join(tmp, "fakerepo"),
    )
    check(True, "a task with no constraint accepts a frame full of duplicates")
    check(
        vr.shard_plan("code_packages", repeated) == [
            {"label": "", "keys": {}, "rows": 2}
        ],
        "and shard_plan reports it as one piece",
    )

    vr._build = real_build

    print("8. one client for every LLM call, and no OpenAI anywhere")
    check(
        vr.TASKS["summarization"].keys == ("TOGETHER_API_KEY",)
        and vr.TASKS["simplification"].keys == ("TOGETHER_API_KEY",),
        "summarization and simplification need no OpenAI key",
    )
    fake = os.path.join(tmp, "fakehalogen")
    os.makedirs(os.path.join(fake, "verifiers"))
    os.makedirs(os.path.join(fake, "src"))
    files = {
        "verifiers/__init__.py": "",
        # Same call shape as verifiers/historical_events.py: builds its own
        # Together client in __init__, then calls completions.create.
        "verifiers/historical_events.py": (
            "import os\nimport pandas as pd\n"
            "Together = None  # stands in for: from together import Together\n"
            "class InterpersonalRelationshipEvaluator:\n"
            "    def __init__(self, api_key, model):\n"
            "        self.client = Together(api_key=api_key)\n"
            "        self.model = model\n"
            "    def evaluate(self, csv_path, output_dir):\n"
            "        df = pd.read_csv(csv_path)\n"
            "        df['verdict'] = [self.client.completions.create(\n"
            "            prompt='Q: ' + r, model=self.model, max_tokens=5).choices[0].text\n"
            "            for r in df.response]\n"
            "        stem = os.path.basename(csv_path)[:-4]\n"
            "        df.to_csv(output_dir + stem + '_AU.csv', index=False)\n"
        ),
        # Same shape as verifiers/get_model_responses.py: "gpt" goes to
        # OpenAI, anything else to a fresh Together client, errors become "".
        "verifiers/get_model_responses.py": (
            "Together = None\n"
            "def get_response_target(prompt, model_name, api_key):\n"
            "    if 'gpt' in model_name:\n"
            "        raise RuntimeError('this would call OpenAI')\n"
            "    client = Together(api_key=api_key)\n"
            "    try:\n"
            "        r = client.chat.completions.create(model=model_name,\n"
            "            messages=[{'role': 'user', 'content': prompt}])\n"
            "        return r.choices[0].message.content\n"
            "    except Exception:\n"
            "        return ''\n"
        ),
        # Same shape as verifiers/summarization.py: star import of the helper
        # as a top level module, writes into the CWD.
        "verifiers/summarization.py": (
            "import pandas as pd\n"
            "from get_model_responses import *\n"
            "class SummarizationEvaluator:\n"
            "    def __init__(self, decomposition_api_key, entailment_api_key,\n"
            "                 decomposition_model='x', entailment_model='x'):\n"
            "        self.dk, self.ek = decomposition_api_key, entailment_api_key\n"
            "        self.dm, self.em = decomposition_model, entailment_model\n"
            "    def evaluate(self, csv_path, output_dir=None):\n"
            "        df = pd.read_csv(csv_path)\n"
            "        rows = []\n"
            "        for passage, resp in zip(df.iloc[:, 0], df.iloc[:, 1]):\n"
            "            atom = get_response_target('split: ' + resp, self.dm, self.dk)\n"
            "            ok = get_response_target('entail: ' + atom, self.em, self.ek)\n"
            "            rows.append({'atom': atom, 'is_supported': ok})\n"
            "        stem = csv_path.split('/')[-1][:-4]\n"
            "        pd.DataFrame(rows).to_csv(stem + '_AU.csv', index=False)\n"
        ),
    }
    for rel, body in files.items():
        with open(os.path.join(fake, rel), "w") as f:
            f.write(body)

    class _Backend:
        def __init__(self, fail_on=None):
            self.sent = []
            self.fail_on = fail_on

        def complete(self, model, prompt, **decode):
            self.sent.append(("completion", model, prompt, decode))
            return "yes"

        def chat(self, model, messages, **decode):
            text = messages[-1]["content"]
            self.sent.append(("chat", model, text, decode))
            if self.fail_on and self.fail_on in text:
                raise ConnectionError("provider down")
            return "atom about " + text if text.startswith("split") else "yes"

    class _DictCache:
        def __init__(self):
            self.d = {}

        def get(self, model, prompt):
            return self.d.get((model, prompt))

        def put(self, model, prompt, text):
            self.d[(model, prompt)] = text

    from hallutrace.llm_client import LLMClient

    he_in = os.path.join(tmp, "he.csv")
    pd.DataFrame({"response": ["r1", "r2"], "name1": ["a", "b"], "name2": ["c", "d"]}).to_csv(
        he_in, index=False
    )
    backend, cache = _Backend(), _DictCache()
    client = LLMClient(backend, cache)
    saved_key = os.environ.pop("TOGETHER_API_KEY", None)
    he_out = os.path.join(tmp, "he_out.csv")
    vr.run_verifier("historical_events", he_in, he_out, repo_root=fake, client=client)
    got = pd.read_csv(he_out)
    check(list(got.verdict) == ["yes", "yes"], "a completions verifier is answered by the client's backend")
    check(len(backend.sent) == 2 and backend.sent[0][3] == {"max_tokens": 5},
          "one backend call per row, with the verifier's own decode settings")
    check(saved_key is None or "TOGETHER_API_KEY" not in os.environ,
          "and no TOGETHER_API_KEY was needed, because the client holds the credential")
    import sys as _sys
    check(_sys.modules["verifiers.historical_events"].Together is None,
          "the module's own Together is restored after the call")
    vr.run_verifier("historical_events", he_in, os.path.join(tmp, "he_out2.csv"),
                    repo_root=fake, client=client)
    check(len(backend.sent) == 2 and client.stats() == {"calls": 2, "cache_hits": 2, "failures": 0},
          "a second run is served from the cache and pays for nothing")
    expect_raises(
        RuntimeError,
        lambda: vr.run_verifier("historical_events", he_in, os.path.join(tmp, "x.csv"), repo_root=fake),
        "without a client the Together key is still required and named",
        substr="TOGETHER_API_KEY",
    )

    sm_in = os.path.join(tmp, "sm.csv")
    pd.DataFrame({"prompt": ["the passage"], "response": ["one claim"]}).to_csv(sm_in, index=False)
    backend = _Backend()
    client = LLMClient(backend, _DictCache())
    vr.run_verifier("summarization", sm_in, os.path.join(tmp, "sm_out.csv"),
                    repo_root=fake, client=client)
    check(os.path.isfile(os.path.join(tmp, "sm_out.csv")),
          "summarization imports and runs: verifiers/ is on sys.path for its star import")
    models = {m for kind, m, _, _ in backend.sent}
    check([k for k, _, _, _ in backend.sent] == ["chat", "chat"] and models == {vr.DEFAULT_VERIFIER_MODEL},
          "decomposition and entailment both go through the client, on the verifier model")
    check(not any("gpt" in m for m in models), "nothing is sent to a gpt model")
    expect_raises(
        ValueError,
        lambda: vr.run_verifier("summarization", sm_in, os.path.join(tmp, "g.csv"), repo_root=fake,
                                model="gpt-3.5-turbo-0125", client=LLMClient(_Backend())),
        "a gpt model for summarization is refused before any call",
        substr="P0-6.1",
    )

    backend = _Backend(fail_on="entail")
    cache = _DictCache()
    client = LLMClient(backend, cache)
    vr.run_verifier("summarization", sm_in, os.path.join(tmp, "sm_fail.csv"),
                    repo_root=fake, client=client)
    check(len(client.failures) == 1 and client.failures[0]["kind"] == "chat",
          "a provider failure HALoGEN swallows into '' is still counted")
    check(pd.read_csv(os.path.join(tmp, "sm_fail.csv")).is_supported.isna().all(),
          "and the verifier still sees it the way it always did, as ''")
    check(len(cache.d) == 1, "only the call that succeeded is cached, never the failure")
    if saved_key is not None:
        os.environ["TOGETHER_API_KEY"] = saved_key
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("all tests passed")
