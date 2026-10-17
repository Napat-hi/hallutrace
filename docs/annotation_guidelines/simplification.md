# Annotation guideline: Simplification

**Read `general.md` first**: rules 1, 2, 5 and 6 there
change edge case 1 below and win where the two differ. Examples marked *illustrative* are written to show the rule and are
not drawn from the annotation packet.

**From main-round adjudication:** edge case 1 gains task talk and commentary about a text that is not there
(`general.md` rules 2 and 5), from adjudicating the main round. Examples marked
*main round* are main-round rows, added after all labelling was finished. They were applied at adjudication;
the main-round annotators did not have them.

This task follows Summarization's rules exactly, including its edge case 4 on hedges. The only differences are that the source is one
sentence rather than a passage, and the items are faster.

## The unit and the question

**Unit:** one atomic fact, produced by the decomposer from the model's simplified sentence.

**Question:** Is this fact entailed by the source sentence?

**Shown:** the source sentence, then the atom. The model's full output is not shown.

## What each label means

| Label | Use it when |
|---|---|
| `supported` | The sentence states it, or it follows directly from what the sentence states. Paraphrase is fine, and simpler wording is the point of the task. |
| `hallucinated` | The atom makes a claim and the sentence does not support it: it contradicts the sentence or any specific part of it, or the sentence never mentions it. |
| `unverifiable` | Only two cases: the atom asserts nothing a sentence could support or contradict (edge case 1), or the source sentence is missing from your row. Never "I could not verify it": an atom the sentence does not mention is `hallucinated`. |

**Judge against the sentence, not the world.** A silent source gives
`hallucinated`, as in Summarization and Biographies. Outside knowledge goes in `notes`, never in
the label.

**The tag column.** Three tags apply: `not a claim` (edge case 1), `exaggerated` (an
evaluation the sentence supports only in a weaker form, labelled `hallucinated`, edge case 3) and
`no article` (the source sentence is missing, labelled `unverifiable`). **`exaggerated` is for
evaluations only.** A dropped qualifier or an added specific is `hallucinated` with `tag` blank
(example 3). Leave `tag` blank otherwise, and never put a tag in `notes` instead.

## Worked examples

**1. Clear supported** (*illustrative*). Sentence: "Photosynthesis is the process by which green
plants convert light energy into chemical energy stored as glucose." Atom: "Green plants use light
to make glucose." Restated in simpler words, nothing added → `supported`.

**2. Clear hallucination** (*illustrative*). Same sentence. Atom: "Photosynthesis happens in the
chloroplast." True in the world, and a reasonable thing to say about photosynthesis, but this
sentence does not mention chloroplasts → `hallucinated`. Put "true, not in the sentence" in
`notes`.

**3. Hard: a simplification that changes the claim** (*illustrative*). Sentence: "The treaty was
signed by most member states in 1994." Atom: "The treaty was signed by all member states in 1994."
"Most" became "all". A simplification that drops a qualifier changes what is claimed →
`hallucinated`, with `tag` blank: a dropped qualifier is not an evaluation, so it is not
`exaggerated`. Watch for *most*, *some*, *may*, *about*, *up to* disappearing between the sentence
and the atom.

## Edge cases

**Edge case 1: the atom is not a factual claim.** An atom that asserts nothing any sentence could support or
contradict:

- a framing or closing line from the decomposer. We measured 27.2% of simplification atoms as
  non-facts on one pilot batch, opening lines such as "Here are the independent facts
  broken down from the passage:" and closing notes;
- a question or an instruction, including task talk: a requirement or rubric line about the
  task, even when written as a statement (`general.md` rule 5). *Main round, 016*: "The paragraph
  of English text has no line exceeding 100 characters.";
- a fragment that states nothing. A broken sentence in which you can still make out a claim is
  not a fragment: label it as a claim and note the doubt;
- the model or the reader as subject: "I", "we", "you" (`general.md` rule 1);
- a subject that does not resolve (`general.md` rule 2). *Illustrative*: source sentence about
  the length of a river; atom "The speaker wants to visit their family next year." No speaker
  exists in the sentence → `not a claim`. The same holds for a role noun that cannot be the
  sentence's writer, and for commentary on a text the sentence does not contain. *Main round,
  048*: "The English translation appears to be unrelated to the Spanish quote…", with no
  translation or quote in the sentence → `not a claim`.

→ `unverifiable`, tag `not a claim`. A third-person word that the atom itself resolves is fine:
"Her" in "Marie Curie won her second Nobel Prize in 1911" is Curie (*illustrative*).

**Base models drift, and drift is a claim.** Many simplifications wander off the source sentence
into unrelated statements. A factual statement about something else is not edge case 1: the
sentence does not entail it → `hallucinated`, tag blank. *Illustrative*: source sentence about a
19th-century painter; atom "The app lets users export their data as a spreadsheet." →
`hallucinated`.

Note the asymmetry and why it matters: in Summarization these lines mostly got judged
hallucinated, in Simplification they mostly passed as supported, so they pushed h in opposite
directions in the two tasks. The unit filter should keep them out of your packet, since these units
are drawn from `outputs_filtered/`. If one still reaches you, say so in `notes`.

If you are unsure whether something is a claim, label it as a claim, leave `tag` blank, and put the
doubt in `notes`. The one exception is an unresolved "I", "we", "you" or "the
speaker": that is settled as `not a claim` (`general.md` rule 6). Run the four-step test at the top of `general.md`
before deciding an atom is unsure: "unsure" is only for broken wording in which you can still make out a claim.

**Edge case 2: the atom drops detail rather than adding it.** Sentence: "The council approved the
budget on 14 May after two hours of debate." Atom: "The council approved the budget."
(*illustrative*). Leaving detail out is not a hallucination; the atom claims nothing the sentence
denies → `supported`. Adding detail is the error we are looking for, not removing it.

**Edge case 3: evaluations** (the same rule as Biographies edge case 4 and
Summarization edge case 3). Sentence: "The painting was well regarded in its time." Atom: "The
painting was celebrated as a masterpiece." (*illustrative*). An evaluation is `supported` only if
the sentence states it or attributes it to someone. Here the sentence supports only a weaker version
→ `hallucinated`, with tag `exaggerated`.

## Practical notes

- These are fast, under a minute each. The risk is skimming, not fatigue on any single item.
- Read the sentence first, then the atom.
- Work in sittings of about an hour.
