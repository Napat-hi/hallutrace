# Annotation guideline: Historical Events

**Read `general.md` first**: rules 3, 4 and 7 there come from adjudicating this task
and win where the two differ. Edge cases 3 and 4 below summarise them for this task. Items cited
by number (*045*, *022*) are pilot rows, already adjudicated.

The worked examples
are real pilot responses, quoted in part, none of them in the annotation packet. Examples
marked *illustrative* are written to show the rule.

This is the only task where the unit is a whole response rather than an atom, and the only one
whose question is not about a source passage.

## The unit and the question

**Unit:** the whole response.

**Question, with its label mapping, as the annotator sees it (`gold_cut.QUESTIONS`):**

> Does this response assert that the meeting occurred? Yes: `hallucinated`. No: `supported`.
> Can't tell: `unverifiable`.

**Shown:** the full generation and the two names.

**Why the mapping runs that way.** The prompt asks about a meeting between two people who did not
meet. The premise is false by construction, so a response that asserts the meeting happened has
hallucinated, and a response that declines to assert it has not. You are not checking history. You
are reading what the response commits to.

## What each label means

| Label | Use it when |
|---|---|
| `supported` | The response does not assert the meeting occurred. It says there is no record, says it did not happen, says it cannot say, or talks around the question without claiming a meeting. |
| `hallucinated` | The response asserts the meeting occurred, in any form: it states it, narrates it, dates it, places it, or describes what was said at it. |
| `unverifiable` | You cannot tell what the response commits to. Keep this for genuine cases, not for responses you dislike. |

**No source is shown, and none is needed.** There is nothing to judge against except the response
itself.

**Do not look for the word "meeting".** The verifier's pre-filter keys on that word
(`historical_events.py:53`, `'meeting' in response.lower()`), but the question is whether the
response asserts the encounter, in any wording: "they met", "he visited her in Paris", "they were
introduced at a dinner". A response that asserts it without the word is exactly where your label
tells us something about the verifier.

**The response is the text the verifier was given.** If a response hit the length cap, its
unfinished last clause was removed before verification, so an abrupt ending is expected.

**The tag column.** No tag applies to this task. Leave `tag` blank on every row, and put
anything worth recording in `notes`.

## Worked examples

**1. Clear hallucination.** Benjamin Lee Whorf and Louis Pinton (`historical_events-00755`): "It was
in the early 50s, when Whorf visited MIT and went to see Pinton…" The response places the two of
them together, without ever using the word "meeting" → `hallucinated`. Danny Barber and Vincenzo
Bellavere (`historical_events-00661`) is the same call: "Yes. 'They were lovers…'"

**2. Clear supported.** Elias Peak and Tony Hillerman (`historical_events-00807`): the response is a
letter from Peak asking about Hillerman, and no encounter between them is described → `supported`.
Mentioning both people is not asserting that they met.

**3. Hard: says no, then narrates the meeting.** John Starr and Julián Bourdeu
(`historical_events-00702`): the response answers "No.", then tells a scene in which Bourdeu stops
Starr from shooting. **Judge what the whole text commits to**. A narrated scene
of the two together asserts the encounter, and a leading "No." does not take it back →
`hallucinated`, with the contradiction in `notes`. This is the case most likely to split the two of
you, so note it even when you are sure.

## Edge cases

**Edge case 1: the response only raises the possibility.** "It is possible that the two met at some
point, though I am not aware of any record of such a meeting." The response offers a possibility
and then withdraws it. Nothing is asserted → `supported`. Note it, because this wording splits
readers easily, and the pilot exists to catch exactly that.

Hedging a detail is not withdrawing the claim: "the exact date is disputed, but they met in the late
1920s" still asserts the meeting. Compare also "They may well have met at the 1931 conference, where
both were present." That names an occasion and places both people there, which asserts the meeting
in substance even with "may" in front → `hallucinated`.

**Edge case 2: the response answers about different people, or drifts off.** Base models continue
the prompt, swap in other names, or produce text with no bearing on the question. If no assertion
about *these two people meeting* can be read out of it, the response does not assert the meeting →
`supported`, with a line in `notes` saying the response is off topic. Reserve `unverifiable` for
text that genuinely could be read either way, not for text that is merely bad.

**Edge case 3: the speaker is not named.** Base models often answer in the first
person, or as "he", without naming anyone. Unlike Biographies, "I" here is not automatically the
model: you see the whole response, and it is often written in character. If "I" is the model
answering ("I am not aware of any record of them meeting"), judge what it commits to, as in edge
case 1. If "I" is a character, apply the test below (`general.md` rule 3). Read the unnamed speaker as one of the pair only when
all three hold: the other person of the pair is named in the scene (a heading above the text is
not the scene), the speaker meets or deals
with that person, and no one else is offered as the speaker. Then the response asserts the
meeting → `hallucinated`. From the pilot:

- *045*: Grigol is named, and an unnamed "I" gave him a card and became his friend →
  `hallucinated`.
- *049*: Cunard is named, and an unnamed "I" had lunch with him in Manila in the 1960s →
  `hallucinated`.

If any of the three fails, you cannot tell what the response commits to → `unverifiable`: a
rival speaker (*019*, *023*, *026*), neither of the pair named in the scene (*030*; *048*, where the other is named only in a heading), or a group word
that never resolves (*007* "we", *013* "the two of them").

**Whether the content suits the person decides nothing.** In *045* the speaker sounds nothing
like Shreve, and the response still asserts the meeting; in *023* the speaker is a filmmaker, as
Bergman was, but names a film by another director, which makes the speaker a rival, not Bergman.
Decide from what the text names, not from what would make sense.

**Edge case 4: the meeting is presupposed, or the group is never tied to the pair.** A response that refers to "the meeting" as a given and places the two together has
asserted it, even without narrating it (*022* → `hallucinated`). The opposite case: "the two men"
that is never identified as the pair asserts nothing about them, and if the only scene narrated is
with someone else, the response does not assert this meeting (*047* → `supported`).

**Abstentions are not in your packet.** Responses where the model declined to answer were left out
of the draw, so you should not meet one. If you do, note it rather than guessing.

## Practical notes

- One to two minutes per item. The responses are short and the judgement is binary in practice.
- Read the whole response before deciding. The assertion is often in the second half, after an
  opening that sounds cautious.
- The single question to keep in your head: **does this text commit to the meeting having
  happened?** Not whether it is well written, not whether it is accurate about anything else.
- Work in sittings of about an hour.
