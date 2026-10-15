# Annotation guideline: Summarization

**Read `general.md` first**: rules 1, 2, 5 and 6 there
change edge case 1 below and win where the two differ. Examples marked *illustrative* are written to show the rule and
are not drawn from the annotation packet.

## The unit and the question

**Unit:** one atomic fact, produced by the decomposer from the model's summary.

**Question:** Is this fact entailed by the source passage?

**Shown:** the source passage, then the atom. The model's full summary is not shown, on purpose:
judge the atom as written, not what the summary may have meant around it.

## What each label means

| Label | Use it when |
|---|---|
| `supported` | The passage states it, or it follows directly from what the passage states. Paraphrase is fine. |
| `hallucinated` | The atom makes a claim and the passage does not support it: the passage contradicts it or any specific part of it, or the passage never mentions it. |
| `unverifiable` | Only two cases: the atom asserts nothing a passage could support or contradict (edge case 1), or the source passage is missing from your row. Never "I could not verify it": an atom the passage does not mention is `hallucinated`. |

**Judge against the passage, not the world.** An atom can be true in the world
and still be `hallucinated` if this passage does not support it. If you know it is true from
elsewhere, say so in `notes` and label it by the passage. **A silent passage gives
`hallucinated`**, the same rule as Biographies: the entailment judge answers True or False on the
context it was given, so κ measures whether it got that right rather than the distance between two
definitions.

**The tag column.** Three tags apply to this task: `not a claim` (edge case 1),
`exaggerated` (an evaluation the passage supports only in a weaker form, labelled `hallucinated`,
edge case 3) and `no article` (the source passage is missing, labelled `unverifiable`).
**`exaggerated` is for evaluations only.** An atom that adds a specific the passage does not support,
or drops a qualifier, is `hallucinated` with `tag` blank. Leave `tag` blank otherwise, and never put a
tag in `notes` instead: the agreement script reads tags from `tag` and stops on any other value there.
`notes` stays free text, so a reason can sit beside a tag.

## Worked examples

**1. Clear supported** (*illustrative*). Passage: "The plant closed in March 2011 after a
twelve-year run, cutting 240 jobs in the county." Atom: "The plant closed in 2011." The passage
states it → `supported`. A less specific atom than the passage is still supported.

**2. Clear hallucination** (*illustrative*). Same passage. Atom: "The plant closed because of
falling demand for its products." The passage gives no reason at all. Plausible, possibly true,
never stated → `hallucinated`.

**3. Hard: one part supported, one part added** (*illustrative*). Same passage. Atom: "The closure
cut 240 jobs across the state." The number is right; "across the state" is not what the passage
says (the county). Any unsupported specific makes the whole atom `hallucinated`, even when the rest
is right. This matches Biographies example 3 and is the same rule in both tasks. It is a wrong
specific, not an evaluation, so `tag` stays blank.

## Edge cases

**Edge case 1: the atom is not a factual claim.** An atom that
asserts nothing any passage could support or contradict. Only five kinds:

- a framing or closing line from the decomposer (below);
- a question or an instruction ("Please paste the article you want summarised.", *illustrative*);
- a fragment that states nothing ("The early years.", *illustrative*). A broken sentence in which
  you can still make out a claim is not a fragment: label it as a claim and note the doubt;
- the model or the reader as subject: "I", "we", "you" (`general.md` rule 1). *Illustrative*:
  "We now list the main points in order.";
- a subject that does not resolve: a "he", "she", "it" or "the speaker" with no one, or more than
  one, it could mean (`general.md` rule 2).

A third-person word **does** resolve when the atom names the person, or when the passage is about
one person. *Illustrative*: passage about one tennis player's comeback; atom "Her withdrawal would
have been a mistake so early in her season." "Her" is the player, so it is a claim (an
evaluation, edge case 3).

**A factual statement about something other than the passage is a claim, not edge case 1.** The
question is whether the passage entails the atom, and it does not → `hallucinated`, tag blank.
This is content the model made up, which is what we are measuring. *Illustrative*: passage about
a football transfer; atom "The Eiffel Tower is in Paris." True, and still → `hallucinated`. The commonest source is the decomposer itself, which opens most decompositions with
a framing line, and Llama-3.3 closes some with a note. We measured 25.2% of summarization atoms as
non-facts on one pilot batch. Real examples from that measurement: "Here are the
independent facts broken down from the passage:", "Note that the passage is mostly repetitive…",
"If you provide a real passage, I'll be happy to help." → `unverifiable`, with tag `not a claim`.

The framing filter should keep these out of your packet, because Summarization units are drawn
from `outputs_filtered/`. If one reaches you anyway, that is a finding worth a note,
not a reason to guess.

If you are unsure whether something is a claim, **label it as a claim** (`supported` or
`hallucinated`, by the passage), leave `tag` blank and put the doubt in `notes`.
The one exception is an unresolved "I", "we", "you" or "the speaker": that is settled as
`not a claim`, not a doubt (`general.md` rule 6). Run the four-step test at the top of `general.md`
before deciding an atom is unsure: "unsure" is only for broken wording in which you can still make out a claim.
Adjudication only sees rows where the two of you differ, so a doubt you both share would otherwise
be lost.

**Edge case 2: the atom carries two facts.** Some atoms hold two claims in one line, for example
"The plant closed in 2011 and its owner moved production abroad" (*illustrative*). **Every part
must be entailed.** If the passage supports one half and not the other, the atom is
`hallucinated`. Do not split it and do not judge the half you prefer.

**Edge case 3: evaluations** (the same rule as Biographies edge case 4). Some atoms
judge rather than state: "The film was a critical success", "The decision was widely criticised"
(*illustrative*). Judge them like any other claim:

- `supported` only if the passage states the evaluation or attributes it to someone ("critics
  praised the film");
- otherwise `hallucinated`;
- if the passage supports only a weaker version (it says the film "was well received in France",
  the atom says "a critical success"), still `hallucinated`, with tag `exaggerated`.

A mixed atom needs every part supported, as in edge case 2.

**A note on what is not an error.** A summary that leaves information out is not hallucinating. We
are only asking whether what the atom says is entailed, never whether the summary was complete.

## Practical notes

- Aim for two to four minutes per item. If you pass five, pick the best label, write why in
  `notes`, and move on. Adjudication settles it.
- Read the passage first, then the atom. Reading the atom first makes it easy to find words in the
  passage that look like support.
- Work in sittings of about an hour.
