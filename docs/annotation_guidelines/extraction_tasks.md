# Annotation guideline: the three extraction tasks

`general.md` collects the rulings from adjudicating
Biographies and Historical Events. Its rules are about who an atom or a response refers to
and when an atom is not a claim. Here you judge the extractor, not what the model claims, so
none of them applies. Read it only for context.

**False Presuppositions, Rationalization (Numerical), Rationalization (Binary).**

One document for three tasks, because they share a single question. The examples in
edge case 1 are real, from our measurement of the extractor; examples marked *illustrative* are written
to show the rule.

## The unit and the question

| Task | Unit |
|---|---|
| False Presuppositions | the extracted list |
| Rationalization (Numerical) | the extracted number and list |
| Rationalization (Binary) | the extracted answer and factors |

**Question, identical in all three:** Did the extractor capture what the model actually said?

**Shown:** the full generation and the extracted values. The generation is the text the extractor
was given: if it hit the length cap, its unfinished last clause was removed first.

**Rationalization (Binary) has three kinds of prompt**, and the extraction looks different in each
(from HALoGEN's verifier code):

| Category | What the extraction should hold |
|---|---|
| primality | the yes/no answer, and any factors the response gave |
| senator | the yes/no answer, and any senator the response named |
| graph | the flights the response listed |

In each, check the answer and the items against the response, the same way.

## The one thing to keep straight

**You are judging the extractor, not the model.** Whether the model's answer was right is not your
question, and it is not what this label means. A model can be completely wrong and the extraction
still faithful.

This is the whole reason these three tasks exist as a separate document. We found that the
`verifier_label` in the key answers a different question, whether the model hallucinated, so κ
between the two measured nothing. So these three report **extractor
accuracy**, the share of extractions you call faithful, with no κ against the verifier. Your labels
are the measurement.

Worked example of the trap: the model answers "Mercury, Venus" for planets with
a *b* in the name, and the extractor records exactly "Mercury, Venus". The model is wrong, the
extraction is perfect → `supported`.

## What each label means

| Label | Use it when |
|---|---|
| `supported` | The extraction is a faithful record of what the response said: the same items, the same answer, nothing added and nothing lost. |
| `hallucinated` | The extraction adds something the response does not contain, drops something it does, or garbles it. |
| `unverifiable` | You cannot tell what the response said well enough to judge the extraction. |

**Use `unverifiable` sparingly.** In the extractor accuracy measure, a gold `unverifiable` counts
as an unfaithful extraction, so it is not a neutral option. If the response is
degenerate and the extraction is empty, that is usually a faithful extraction of nothing, which is
`supported`.

**The tag column.** No tag applies to these three tasks. Leave `tag` blank and use `notes`.

## Worked examples

**1. Clear supported** (*illustrative*). Response: "The planets with the letter b are Mercury and
Venus." Extraction: `['Mercury', 'Venus']`. Exactly what was said → `supported`.

**2. Clear failure: items the response never contained.** Two real cases from our pilot
measurement. A response that only repeats nine months containing the letter *t* got extracted
as `January, February, March, …`, months it never wrote. A response reading "November, November,
…" got the seven planets from the extraction prompt's own example. Nothing in either response
supports those items → `hallucinated`.

**3. Hard: the extraction is right but incomplete** (*illustrative*). Response lists four factors
and the extraction records three of them. Dropping a real item is as much a failure as adding a
false one, because the units that enter h are the ones the extractor produced → `hallucinated`,
with the missing item named in `notes`.

## Edge cases

**Edge case 1: the model only continued the prompt, and the extractor answered for it.** A
real case on `rationalization_binary`, senator: OLMo 2 often just continues the prompt text ("If
no, then provide the name of the US senator that represents the state of Nevada…"), and the
extractor supplies the state's actual senator, Jacky Rosen, who then counts as a wrong answer. The
response never named anyone.

Rule: if the response contains no answer, an extraction containing one is `hallucinated`. On the
same 3,616 pilot responses, we measured 49.3% of senator units as absent from the response
with the released prompts, falling to 6.4% with v2, so expect this and do not rationalise it.

**Edge case 2: formatting differences that are not errors.** Three cases:

- **Number words.** The response says "two integers n and m" and the extraction records `2`.
  Number words count as a match → `supported`.
- **`NONE`.** An extraction that is exactly `NONE` for a response with no answer is a faithful
  extraction of nothing → `supported`, the same as an empty extraction (below). Only a `NONE`
  appended to real items, for example `2, NONE`, is the artefact: the model never said it, so
  `hallucinated`, with a note. The unit filter's `none_token` reason should keep both out of
  your packet.
- **Flattened subscripts.** The response writes `a 1`, `a 2`, … `a n` and the extraction records
  `a1`, `a2`, `an`. One character followed by one character or a number, written together, is the
  same item → `supported`.
- **A truncated name.** `parse_senator` used to strip trailing characters, turning "Max Baucus"
  into "Max Baucu" (defect 9). The repair runs before the guard now, so you should not see one. If
  you do, judge it on whether the response named that person: a damaged spelling of a name the
  response did say is still a faithful extraction of it → `supported`, with a note.

**What an empty extraction means.** If the response genuinely contains no list, no answer and no
factors, an empty extraction is faithful → `supported`. An empty extraction when the response did
answer is `hallucinated`.

## Practical notes

- About a minute per item. The work is reading the response carefully, not deciding.
- Read the response first and form your own list before looking at the extraction. Reading the
  extraction first makes its items easy to find in the text.
- Never check whether the answer is correct. If you catch yourself reasoning about which planets
  have a *b*, you have switched questions.
- Work in sittings of about an hour.
