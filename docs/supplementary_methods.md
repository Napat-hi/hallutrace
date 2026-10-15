# Supplementary methods: benchmark, models, generation and verification

This is the long form of the paper's setup section. The paper's own text is shorter; where they
differ in detail, this file has it. Every change to HALoGEN's code is in `docs/DEVIATIONS.md`.

## Benchmark and sample

We build on HALoGEN (Ravichander et al., 2025), pinned at commit `8b7dd77e`. We use eight of its
nine tasks: Biographies, False Presuppositions, Historical Events, Rationalization (Binary),
Rationalization (Numerical), Scientific Attribution, Simplification and Summarization. We
exclude code generation. Where the released CSVs differ from the paper, we treat the CSVs as
authoritative. The eight tasks release 9,839 prompts.

We draw 400 prompts per task, 3,200 in total, with a fixed seed (20260916). Where a task's
released data carries a label column, we stratify on it with proportional allocation:
Rationalization (Binary) on `category` (graph, primality, senator), and False Presuppositions
and Rationalization (Numerical) on `condition` (starts with, contains, ends with). The other
five tasks are simple random draws. Scientific Attribution cannot be stratified by source: its
prompt file carries no source label, so Hetionet, SciFact, TruthfulQA and COVID-19 Lies appear
in unknown proportion. The selected prompt ids are in `data/prompt_ids.csv`, and
`scripts/draw_sample.py` redraws the sample from HALoGEN's files.

## Subject models

We study five open 7B-scale base models, chosen because each has a searchable pretraining
corpus. Each model is pinned to the Hugging Face commit its weights were downloaded at.

| Model | Hugging Face repo | Revision | Pretraining corpus |
|---|---|---|---|
| OLMo 2 7B | `allenai/OLMo-2-1124-7B` | `7df9a82` | OLMo-mix-1124 |
| DCLM-7B | `apple/DCLM-7B` | `c85bfa1` | DCLM-Baseline |
| OLMo 1.7 7B | `allenai/OLMo-7B-0424-hf` | `e3c82a0` | Dolma v1.7 |
| RedPajama-INCITE 7B | `togethercomputer/RedPajama-INCITE-7B-Base` | `78f7e48` | RedPajama |
| Pythia 6.9B | `EleutherAI/pythia-6.9b` | `c0e3eee` | The Pile |

All five are base models, not instruction-tuned. Every model HALoGEN evaluated was
instruction-tuned, so our absolute rates are not directly comparable to HALoGEN's. Three of the
models are published in fp32 (OLMo 2, OLMo 1.7, DCLM); we run all five in bfloat16.

## Generation

For each prompt and model we draw one greedy generation (temperature 0) and five sampled
generations (temperature 1.0, nucleus p = 0.9): 3,200 × 5 × 6 = 96,000 generations. All
generations share one prompt format and one length budget across models and tasks:

- **Prompt:** HALoGEN's `prompt` field verbatim, with no chat template or added instruction.
- **Length:** at most 128 new tokens, in each model's own tokenizer, with no stop sequences.

HALoGEN's paper states no generation settings. Its code sets them on one path only: GPT chat
models are called with temperature 0 and at most 128 tokens, while every open model ran on the
API provider's defaults at the time, which are not recorded. For the open models, the released
responses are the only evidence, and they bear on length alone: near each model's maximum
length, 76–92% of responses end mid-sentence, so output was cut at a fixed token budget with no
stop sequences (`scripts/halogen_response_lengths.py`). We use 128 tokens because it is the
value in HALoGEN's code, it matches the budget of the OLMo-7B model HALoGEN evaluated (the
closest to ours), and it bounds the repetition base models produce. The sampling settings do
not come from HALoGEN, which generated once per prompt. Temperature 1.0 follows both
SelfCheckGPT (Manakul et al., 2023) and semantic entropy (Farquhar et al., 2024), and p = 0.9
follows semantic entropy. We follow semantic entropy's temperature and nucleus settings only:
it draws ten samples with top-K 50 and adds a reference generation at temperature 0.1, where we
draw five, without top-K, and use a greedy reference, which is exactly reproducible. The earlier
version of semantic entropy found an intermediate temperature of 0.5 best for its own detection
task (Kuhn et al., 2023); we measure stability across samples rather than detect
hallucinations, and take the published setting of 1.0.

Base models rarely stop on their own. Across all 96,000 generations, 75.5% reached the 128-token
cap: 48.1% for OLMo 2 and 50.8% for OLMo 1.7, against 86.7% for DCLM-7B, 92.1% for Pythia and
100% for RedPajama. A response that reached the cap usually stops mid-clause: of the non-empty
responses, 47% (OLMo 1.7) to 90% (RedPajama) hit the cap and end mid-clause. A cut final clause
would be decomposed into a claim the verifier marks unsupported. Assuming one claim per
sentence, that alone would inflate the hallucination rate by 8.3% (OLMo 2) to 18.4% (Pythia), a
spread of 10.1 points that does not cancel when models are compared. We therefore drop the
trailing clause of a response that hit the cap, before decomposition, and store the dropped
text with the generation. A response that stopped on its own is left whole, since its last line
is finished even without a full stop, and on the answer tasks it is usually the answer ("yes",
"3"). A clause counts as finished if it ends in terminal punctuation or a line break, since base
models often answer list prompts one item per line with no full stop. A cut response with no
finished clause loses all its text this way: 0.3% (OLMo 2, RedPajama) to 1.3% (OLMo 1.7) of
non-empty responses (`scripts/generation_quality.py results/generations`). Such responses,
responses that were empty to begin with, and abstentions are left out of the denominator of
the hallucination rate rather than scored as correct, as HALoGEN does; each model's response
ratio, the share of samples left in, is reported beside it (`results/supplementary/`).

Four models are served with vLLM 0.11.0. DCLM-7B cannot be loaded by vLLM, because its
architecture (`OpenLMModel`) has no registered implementation, so we run it with open_lm and
Hugging Face Transformers, one prompt at a time: open_lm ignores the attention mask, and
left-padded batches corrupted 7 of 8 shorter prompts in a check (`modal run
modal_app.py::padding_check`). All generation ran on single NVIDIA H100 80GB GPUs on Modal.

We record an empty response as a separate outcome from an abstention and from a response with
no checkable claim. Empty responses are 0% (RedPajama) to 6.9% (OLMo 2) of each model's
generations, 2.7% overall; abstentions, detected with HALoGEN's phrase list as corrected (see
Verification), are 1.4% (OLMo 2) to 5.8% (Pythia).

## Reproducibility

Every generation's seed and batch are fixed before any GPU runs.

- **Seeds:** derived from the prompt id, model and sample index by hashing, and stored as
  integers.
- **Batches:** fixed size 64, ordered by sample index and then prompt id. The plan is replayed
  in order.

Greedy decoding is reproducible only for a fixed batch composition. For three of the four
vLLM-served models, the same prompt produced different greedy output in a different batch, and
vLLM 0.11.0 has no working batch-invariant mode for them. Our reproducibility claim is
therefore: the pinned weights, the prompt, the stored seed and the recorded batch reproduce the
output. DCLM-7B, generated one prompt at a time, does not depend on batch composition.

Each generation record stores the seed, batch id, decoding settings, a hash of those settings,
the model revision, the serving engine and version, and the GPU type
(`results/generations/`; the plan is in `data/generation_plan/`). Every software environment is
frozen to a lockfile (`locks/`): the local environment and each container image.
`scripts/validate_a2.py` checks a completed run against the plan: every batch present, every
record loadable into the analysis store, every record stamped with the pinned revision.

Verification is reproducible through caches rather than reruns. Every verifier call (the
Together AI decomposer and judge, Semantic Scholar lookups, and OpenFActScore's Gemma prompts)
was stored with its answer, so rescoring makes no new calls. This matters most for
Biographies, whose verifier samples. The verifiers' inputs and outputs are in
`results/verifier/`.

## Verification

We reuse HALoGEN's task-specific decomposers and verifiers, reproducing every prompt, few-shot
example and parsing rule exactly, typos included. We change them only where the released
pipeline cannot run or is demonstrably broken. Each change is listed with its class in
`docs/DEVIATIONS.md`.

**Retired models.** HALoGEN's closed models are retired or retiring: gpt-3.5-turbo-0125 retires
on 23 October 2026, as does the `gpt-3.5-turbo` behind the validator of FActScore (Min et al.,
2023); FActScore's generator, InstructGPT, was shut down in January 2024. We therefore
standardise all decomposition and entailment calls on Llama-3.3-70B-Instruct-Turbo via Together
AI. The released code mixed Llama 3.1 and 3.3.

**Biographies.** We replace FActScore with OpenFActScore (Lage and Ostermann, 2025), an open
reimplementation of both stages. It reports a Pearson correlation of 0.99 with the original,
but it scores consistently lower in absolute terms, so our Biographies numbers are not
comparable to HALoGEN's. OpenFActScore's authors recommend OLMo 2 as the atomic-fact
generator. We do not use it, because OLMo 2 7B is one of our subject models and a verifier from
the subject's own family would not be independent. We use Gemma 3 4B (instruction-tuned) as
both the atomic-fact generator and the validator. It was the best-performing model in both
roles in the OpenFActScore evaluation (validator cumulative error rate 12.2 against 13.4 for
Llama 3.1; generator BERTScore F1 0.867, the highest average), and it runs on a single GPU
alongside the retrieval index. Using one model in both roles means the two stages may share
errors; the human agreement reported for Biographies (paper, Section 5) bounds this. That
agreement covers retrieval as well as judgement: the validator sees five passages retrieved
from the subject's Wikipedia article, while annotators read the whole article. We verify all
six generations per prompt; the greedy one is the only generation directly comparable to
HALoGEN's.

OpenFActScore samples in both stages, as the original FActScore did (its OpenAI calls used
temperature 0.7), and sets no seed. We keep this as released. Two runs over the same 50
responses shared only 59% of their atomic facts (mean per-response F1) and none of their fact
lists exactly, yet produced 21.5 and 21.1 facts per response, agreed on 99.0% of verdicts for
the facts both runs produced, and gave FActScores of 0.230 and 0.229
(`results/bio_batch_check.json`). Biographies results are therefore reproducible at the level
of rates, not of individual atomic facts. The atoms and verdicts we report are released as
files. To cut cost we send each response's Gemma prompts in one batch instead of one at a time,
which runs 1.42× faster and changes nothing else in either stage. Because both stages sample,
we judged it against unbatched-versus-unbatched variation on the same 50 responses, with marks
set before the batched results were in: batching kept the fact overlap (F1 0.604 against 0.592
for two unbatched runs) and the number of facts per response (21.4 against 21.3), and its
verdicts agreed with an unbatched run on 98.98% of the same 1,076 facts, against 98.88% for a
second unbatched validator (deviation 16).

The fact generator also writes past its prompt. After breaking down the sentence it was given,
Gemma often continues in the few-shot format, with a new "Please breakdown the following
sentence" line, a sentence it made up and that sentence's facts. These invented facts are
checked against the subject's Wikipedia page and mostly come back unsupported. The behaviour is
OpenFActScore's own and occurs at the same rate without batching. We drop them after
verification, from the generator's raw outputs, which the cache keeps: any atom that contains
the leaked line, any atom from the text after it, and atoms parsed from a refusal. On the full
run this removed 9,713 of 210,354 atoms (4.6%; 3.8% to 5.6% by model) and moved the unsupported
share by 0.6 to 1.4 points per model, without changing the models' order
(`results/bio_filter_generations.json`). The filter has a cost: a leaked line is often fused
with the real fact before it into one atom, so dropping these 2,198 atoms also drops about 2,198
real facts, and 236 of the dropped atoms had been judged supported. We report Biographies both
with and without this filter, and every dropped atom is listed with its row (deviation 20).

**Extraction tasks.** Four tasks extract units from the response with a few-shot or zero-shot
prompt before checking them: False Presuppositions, the three Rationalization (Binary)
categories, Rationalization (Numerical) and Scientific Attribution. With the released prompts,
Llama 3.3 returned units that do not appear in the response: it answered from its own
knowledge or copied a few-shot example when the response had nothing to extract. Setting its
temperature to 0 did not change the rate (15.6% against 16.3%), so the cause is the prompts.
The Summarization and Simplification decomposers also emitted their own framing lines ("Here
are the independent facts…") as facts, 25–27% of units on one pilot batch. To the four
extraction prompts we add a negative few-shot example answering `NONE` and an instruction to
use only text that appears in the response; the decomposer's prompt is kept as released. After
extraction we drop any unit not found in the response and the decomposer's framing lines, list
every dropped unit next to its row, and report the hallucination rate both with and without
this filter. On the pilot responses, the share of extracted units not found in the response
fell from 35.8% to 15.5% (False Presuppositions), 43.0% to 10.1% (Rationalization
(Numerical)), 29.5% to 7.2% (Scientific Attribution) and 49.3% to 6.4% (the senator category).
In the full run the filter dropped 15.4% of extracted units on False Presuppositions, 8.3% on
Rationalization (Numerical), 9.6% on Rationalization (Binary) and 5.6% on Scientific
Attribution, mostly units not found in the response, and 18.4% on Summarization and 21.8% on
Simplification, mostly framing lines (deviations 17 to 19; `scripts/posthoc_filter.py` on
`results/verifier/generations-prompts-v2/outputs`).

**Fixes that change measured rates.**

- HALoGEN's abstention detector has a missing comma that merges "I am sorry" and "Sorry" into
  one string that never matches. We fix it and add "I cannot" and "I am unable", so our
  abstention rates are not directly comparable to HALoGEN's. The Scientific Attribution
  verifier kept a second, unfixed copy of the list; we replaced both with one shared list.
- A failed Semantic Scholar lookup counted as a hallucination in the released code. We record
  it as `unverifiable`, a third verdict alongside `supported` and `hallucinated`. Lookups are
  paced to Semantic Scholar's limit of one request per second and retried with exponential
  backoff.

**Validation.** Because the verifier configurations changed, none of them is validated by
HALoGEN's published agreement figures. We validate each task against human annotation before
its results enter the analysis (paper, Section 5; `annotation/`).

## Compute

| Stage | Hardware | Measured cost |
|---|---|---|
| Generation, four vLLM models | H100 | ≈0.05 s per generation at batch 64 |
| Generation, DCLM-7B | H100, 8 containers in parallel | 1.5–2.1 s per generation |
| Biographies verification | H100, Gemma 3 4B, up to 10 in parallel | 11.7–12.6 s per response unbatched, 8.2 s batched (the 50-response check); full run of 10,856 responses: $142, 3 h 40 min plus a 20 min redo |
| Other seven tasks | Together AI API, CPU containers | run of the six tasks other than Biographies and Historical Events: 1,260 work units, 0 failures, $65.61 on Together and about $0.55 on Modal |

Generation cost about $50 on Modal and took about 1 h 40 min of wall-clock time, with at most
ten H100s running at once. Biographies cost about $142 on Modal: one run ($133), seven units
verified earlier (about $4), and a redo of the 663 responses the cap-only tail rule changed
($5). Together AI cost $72.77 in all: $65.61 for the run of the six tasks and $7.16 before it,
for pilots and the first runs.

## References

- Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. Nature 630, 625–630. doi:10.1038/s41586-024-07421-0
- Kuhn, L., Gal, Y., and Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. ICLR 2023. arXiv:2302.09664
- Lage, L. F., and Ostermann, S. (2025). OpenFActScore: Open-Source Atomic Evaluation of Factuality in Text Generation. arXiv:2507.05965
- Manakul, P., Liusie, A., and Gales, M. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. EMNLP 2023, Singapore, pp. 9004–9017. doi:10.18653/v1/2023.emnlp-main.557
- Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., and Hajishirzi, H. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. EMNLP 2023, Singapore, pp. 12076–12100. doi:10.18653/v1/2023.emnlp-main.741
- Ravichander, A., Ghela, S., Wadden, D., and Choi, Y. (2025). HALoGEN: Fantastic LLM Hallucinations and Where to Find Them. ACL 2025 (Vol. 1: Long Papers), Vienna, pp. 1402–1425. doi:10.18653/v1/2025.acl-long.71
