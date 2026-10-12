"""Batch Gemma's prompts inside OpenFActScore (pinned commit 35c08f1).

OpenFActScore sends Gemma one prompt at a time. `enable()` patches, in the calling
process only:

- `HFmodel.generate_many(prompts)`: one left-padded `model.generate` per chunk, with
  every setting of `HFmodel._generate` kept (chat template, max_new_tokens,
  do_sample=True, first-token scores) and the same cache keys, so batched and
  unbatched runs share a cache.
- `AtomicFactGenerator.get_init_atomic_facts_from_sentence` and `FactScorer._get_score`:
  copies of the pinned code in which the per-prompt `self.lm.generate` loop becomes one
  `generate_many` call per response (marked CHANGED).

When retrieval finds no passage for an atom, both versions return None for the
response; the original has generated verdicts for the earlier atoms by then, this one
has not. parity_check() compares the two with stub models.

The two copied methods come from OpenFActScore and FActScore (MIT License); see
THIRD_PARTY_NOTICES.md.
"""
import re
import string

import numpy as np

# tokens per padded chunk (prompt tokens x rows), about 40 validator prompts
TOKEN_BUDGET = 64_000
# prompts per chunk; None means only TOKEN_BUDGET limits it (set by bio_batch_sizes)
MAX_ROWS = None

_enabled = False


def _generate_batch(lm, prompts, max_output_length=128):
    """HFmodel._generate for a list, padded on the left. Returns [(text, first_token_scores)]."""
    import torch

    if lm.logits is False and lm.mode == "afv":
        max_output_length = 3
    if lm.logits:
        # Forced True/False uses beam search over force_words_ids; keep the original path.
        return [lm._generate(p, max_output_length=max_output_length) for p in prompts]

    texts = lm.chat_formatter(prompts) if lm.use_chat else list(prompts)
    ids = lm.tokenizer(texts).input_ids  # same call as the original, no padding yet
    pad = lm.tokenizer.pad_token_id
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    results = [None] * len(ids)

    start = 0
    while start < len(order):
        end = start + 1
        while (end < len(order) and len(ids[order[end]]) * (end - start + 1) <= TOKEN_BUDGET
               and (MAX_ROWS is None or end - start < MAX_ROWS)):
            end += 1
        chunk = order[start:end]
        width = max(len(ids[i]) for i in chunk)
        input_ids = torch.LongTensor([[pad] * (width - len(ids[i])) + ids[i] for i in chunk]).cuda()
        mask = torch.LongTensor([[0] * (width - len(ids[i])) + [1] * len(ids[i]) for i in chunk]).cuda()
        out = lm.model.generate(
            input_ids,
            attention_mask=mask,
            max_new_tokens=max_output_length,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=1,
            do_sample=True,
        )
        for row, i in enumerate(chunk):
            text = lm.tokenizer.decode(out["sequences"][row, width:], skip_special_tokens=True)
            results[i] = (text, out["scores"][0][row].detach().cpu().numpy())
        start = end
    return results


def generate_many(self, prompts, max_output_length=128):
    """LM.generate for a list: answers from the cache, generates the rest in batches."""
    keys, todo = [], []
    for p in prompts:
        p = p.strip()  # LM.generate strips too: a prompt must not end in whitespace
        key = f"{p}_0"
        keys.append(key)
        if key not in self.cache_dict and p not in todo:
            todo.append(p)
    if todo:
        if self.model is None:
            self.load_model()
        for p, generated in zip(todo, _generate_batch(self, todo, max_output_length)):
            self.cache_dict[f"{p}_0"] = generated
            self.add_n += 1
    return [self.cache_dict[k] for k in keys]


def get_init_atomic_facts_from_sentence(self, sentences, cost_estimate=None):
    """Copy of the pinned AtomicFactGenerator method; one generate_many per response."""
    from factscore.atomic_facts import best_demos, text_to_sentences

    is_bio = self.is_bio
    demons = self.demons

    k = 1 if is_bio else 0
    n = 7 if is_bio else 8

    prompts = []
    prompt_to_sent = {}
    atoms = {}
    for sentence in sentences:
        if sentence in atoms:
            continue
        top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
        prompt = ""

        for match in top_machings:
            prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(match)
            for fact in demons[match]:
                prompt = prompt + "- {}\n".format(fact)
            prompt = prompt + "\n"

        if self.lm.model_name == "InstructGPT":
            for i in range(n):
                prompt = prompt + f"Please breakdown the following sentence into independent facts: {list(demons.keys())[i]}\n"
                for fact in demons[list(demons.keys())[i]]:
                    prompt = prompt + f"- {fact}\n"
                prompt = prompt + "\n"

        prompt = prompt + f"Please breakdown the following sentence into independent facts: {sentence}\n"
        prompts.append(prompt)
        prompt_to_sent[prompt] = sentence

    if cost_estimate:
        total_words_estimate = 0
        for prompt in prompts:
            if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.lm.cache_dict:
                continue
            total_words_estimate += len(prompt.split())
        return total_words_estimate
    else:
        # CHANGED: one batched call instead of self.lm.generate(prompt) per prompt.
        for prompt, (output, _) in zip(prompts, self.lm.generate_many(prompts)):
            output = output.replace("<|eot_id|>", "")
            output = re.sub(r'-\s*\n', '', output)
            atoms[prompt_to_sent[prompt]] = text_to_sentences(output)

        for key, value in demons.items():
            if key not in atoms:
                atoms[key] = value

        return atoms


def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
    """Copy of the pinned FactScorer._get_score; prompts first, one generate_many, then parsing."""
    decisions = []
    total_words = 0

    # CHANGED: build every prompt first (same text as the original), then generate once.
    prompts = []
    for atom in atomic_facts:
        atom = atom.strip()
        if self.lm:
            self.logger.debug("Retrieving passages from %s", knowledge_source)
            passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
            if not passages:
                return None
            definition = "Answer the question about {} based on the given context.\n\n".format(topic)
            context = ""
            for psg_idx, psg in enumerate(reversed(passages)):
                context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
            definition += context.strip()
            if definition[-1] not in string.punctuation:
                definition += "."
            prompt = "{}\n\nInput: {} True or False?\nAnswer:".format(definition.strip(), atom.strip())

            if cost_estimate:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                    total_words += len(prompt.split())
                elif cost_estimate == "ignore_cache":
                    total_words += len(prompt.split())
                continue
            prompts.append(prompt)
        else:
            prompts.append(None)

    if cost_estimate:
        return total_words

    wanted = [p for p in prompts if p is not None]
    outputs = iter(self.lm.generate_many(wanted)) if (self.lm and wanted) else iter(())

    for atom, prompt in zip(atomic_facts, prompts):
        atom = atom.strip()
        if self.lm:
            output = next(outputs)

            if isinstance(output[1], np.ndarray) and self.lm.logits:
                # TODO: assert with tokenizer vocab len
                logits = np.array(output[1])
                # when logits are available,
                true_score = logits[self.lm.true_id]
                false_score = logits[self.lm.false_id]
                is_supported = true_score > false_score
                is_supported = is_supported.item()
                self.logger.debug("-------------------")
                self.logger.debug(f"Prompt: {prompt}")
                self.logger.debug(f'\nLogits:\nTrue: {true_score}\nFalse: {false_score}\nis_supported: {is_supported}')
                self.logger.debug(f'Output: {output[0]}')
                self.logger.debug("-------------------")
            else:
                # when logits are unavailable
                generated_answer = output[0].lower()
                if "true" in generated_answer or "false" in generated_answer:
                    if "true" in generated_answer and "false" not in generated_answer:
                        is_supported = True
                    elif "false" in generated_answer and "true" not in generated_answer:
                        is_supported = False
                    else:
                        is_supported = generated_answer.index("true") > generated_answer.index("false")
                else:
                    is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

        else:
            is_supported = True

        if is_supported and "npm" in self.model_name:
            npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
            is_supported = npprob > 0.3

        decisions.append({"atom": atom, "is_supported": is_supported})
    return decisions


class _StubLM:
    """Deterministic stand-in for HFmodel: answers from a hash of the prompt, logs prompts."""

    def __init__(self, mode):
        self.mode, self.model_name, self.logits = mode, "stub", False
        self.cache_dict, self.calls = {}, []

    def _answer(self, prompt):
        import hashlib
        h = int(hashlib.sha256(prompt.strip().encode()).hexdigest(), 16)
        if self.mode == "afv":  # all three parsing branches: true, false, neither
            return (["True", "False", "I cannot tell"][h % 3], None)
        return (f"- Fact {h % 97} one.\n- Fact {h % 89} two\n", None)

    def generate(self, prompt, sample_idx=0):
        self.calls.append(prompt.strip())
        return self._answer(prompt)

    def generate_many(self, prompts, max_output_length=128):
        return [self.generate(p) for p in prompts]


class _StubRetrieval:
    def get_passages(self, topic, atom, k=5):
        if "NOPASSAGE" in atom:
            return []
        return [{"title": f"{topic} {i}", "text": f"<s>{atom} passage {i}</s>"} for i in range(k)]


def parity_check(rows, demon_dir):
    """Original vs patched code on the same inputs, with stub models. No GPU, no Gemma.

    rows: [{"name", "response"}]. Returns counts; raises AssertionError on any difference
    in prompts sent, atomic facts, or verdicts.
    """
    import logging

    from factscore.atomic_facts import AtomicFactGenerator
    from factscore.factscorer import FactScorer

    orig_afg = AtomicFactGenerator.get_init_atomic_facts_from_sentence
    orig_afv = FactScorer._get_score

    gen = AtomicFactGenerator.__new__(AtomicFactGenerator)  # skip __init__: no model load
    import json, os, spacy
    from rank_bm25 import BM25Okapi
    gen.nlp, gen.is_bio = spacy.load("en_core_web_sm"), True
    gen.logger = logging.getLogger("parity")
    with open(os.path.join(demon_dir, "demons.json"), encoding="utf-8") as f:
        gen.demons = json.load(f)
    gen.bm25 = BM25Okapi([d.split(" ") for d in gen.demons.keys()])

    scorer = FactScorer.__new__(FactScorer)
    scorer.retrieval, scorer.npm, scorer.model_name = {"ks": _StubRetrieval()}, {}, "gemma+gemma"
    scorer.logger = logging.getLogger("parity")

    n_facts = 0
    for row in rows:
        results = []
        for afg, afv in ((orig_afg, orig_afv), (get_init_atomic_facts_from_sentence, _get_score)):
            AtomicFactGenerator.get_init_atomic_facts_from_sentence = afg
            gen.lm, scorer.lm = _StubLM("afg"), _StubLM("afv")
            pairs, breaks = gen.run(row["response"])
            facts = [f for _, fs in pairs for f in fs]
            decisions = afv(scorer, row["name"], row["response"], facts, "ks")
            afv_calls = scorer.lm.calls
            scorer.lm = _StubLM("afv")
            no_passage = afv(scorer, row["name"], row["response"], facts + ["NOPASSAGE x"], "ks")
            results.append((pairs, breaks, gen.lm.calls, decisions, afv_calls, no_passage))
        AtomicFactGenerator.get_init_atomic_facts_from_sentence = orig_afg
        (p0, b0, g0, d0, v0, n0), (p1, b1, g1, d1, v1, n1) = results
        assert (p0, b0) == (p1, b1), f"atomic facts differ for {row['name']}"
        assert sorted(set(g0)) == sorted(set(g1)), f"AFG prompts differ for {row['name']}"
        assert sorted(set(v0)) == sorted(set(v1)), f"AFV prompts differ for {row['name']}"
        assert d0 == d1, f"verdicts differ for {row['name']}"
        # both return None; only the original generates for atoms before the failing one
        assert n0 is None and n1 is None, f"no-passage handling differs for {row['name']}"
        n_facts += len(d0 or [])
    return {"responses": len(rows), "facts": n_facts}


_originals = {}


def enable():
    """Patch OpenFActScore in this process. Idempotent."""
    global _enabled
    if _enabled:
        return
    from factscore.HFmodel import HFmodel
    from factscore.atomic_facts import AtomicFactGenerator
    from factscore.factscorer import FactScorer

    _originals.setdefault("afg", AtomicFactGenerator.get_init_atomic_facts_from_sentence)
    _originals.setdefault("afv", FactScorer._get_score)
    HFmodel.generate_many = generate_many
    AtomicFactGenerator.get_init_atomic_facts_from_sentence = get_init_atomic_facts_from_sentence
    FactScorer._get_score = _get_score
    _enabled = True


def disable():
    """Put the pinned OpenFActScore code back (for the batched-vs-unbatched check)."""
    global _enabled
    if not _enabled:
        return
    from factscore.atomic_facts import AtomicFactGenerator
    from factscore.factscorer import FactScorer

    AtomicFactGenerator.get_init_atomic_facts_from_sentence = _originals["afg"]
    FactScorer._get_score = _originals["afv"]
    _enabled = False
