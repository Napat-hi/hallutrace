# Annotation guideline: Scientific Attribution

**Three rulings from adjudicating the main round.** Edge case 2 is confirmed
with a main-round example, and edge cases 3 (a venue is not a work) and 4 (a generic title) are
new. They were applied at adjudication;
the main-round annotators did not have them.

`general.md` is about who an atom or a response refers to and when an atom is not a claim. Here
the unit is an extracted reference, so none of its rules applies. Read it only for context.

Examples marked
*illustrative* are written to show the rule and are not drawn from the annotation packet. The invented
titles in edge case 1 are real, from our measurement of the extractor.

This task's verifier extracts references first, but its question covers both halves at once, which is why it
keeps a verifier κ while the other three extraction tasks moved to extractor accuracy.

## The unit and the question

**Unit:** one extracted reference.

**Question:** Is this a real paper, and did the model actually cite it in this response?

**Shown:** the full generation and the extracted title.

**Two things are being asked, and both must hold.** The paper has to exist, and the model has to
have cited it in the response in front of you. Either one failing makes the unit `hallucinated`.

## What each label means

| Label | Use it when |
|---|---|
| `supported` | The paper is real **and** the response cites it. |
| `hallucinated` | Either half fails: the paper does not exist, or the response never cited it. |
| `unverifiable` | You searched and genuinely cannot establish whether the paper exists. |

This is the one task whose verifier can also answer `unverifiable`, because Semantic Scholar
lookups can fail and our fix returns UNVERIFIABLE on a failed lookup rather than an empty string.
Its confusion matrix is therefore 3 × 3 rather than 2 × 3.

**The tag column.** No tag applies to this task. Leave `tag` blank and use `notes`.

**The response is the text the verifier was given.** If a response hit the length cap, its
unfinished last clause was removed first, so an abrupt ending is expected and a reference
cut off there is not part of what you judge.

## How to check existence

Search Semantic Scholar first. If it finds nothing, try Google
Scholar before deciding, and say in `notes` which one found it. Both of you use this order, so you
are searching the same way. A paper counts as real if you find it under
that title, or under a title differing only in punctuation, capitalisation or a subtitle. Give it
about a minute. If a focused search turns up nothing, that is `hallucinated`, not `unverifiable`.
Keep `unverifiable` for the case where you find something ambiguous, for example a title that
exists as a chapter, a preprint with a different name, or a result you cannot confirm is the same
work.

Put the search outcome in `notes` whenever you hesitated. Those notes are what make adjudication
fast.

## Worked examples

**1. Clear supported** (*illustrative*). Response: "…as shown in *Attention Is All You Need*
(Vaswani et al., 2017), the transformer architecture…". Extracted title: "Attention Is All You
Need". Real paper, and the response cites it → `supported`.

**2. Clear hallucination** (*illustrative*). Response cites "*Deep Residual Learning for Temporal
Graph Alignment* (Chen and Watanabe, 2019)". No such paper is findable. The model wrote a
plausible-sounding title → `hallucinated`.

**3. Hard: real paper, not cited here.** The extracted title is a real, well known paper, but
reading the response you find it never mentions it. The extractor supplied the reference from its
own knowledge rather than from the text. → `hallucinated`, because the second half of the question
fails. Say in `notes` that the paper is real and the citation is absent, since that distinction is
what the extractor accuracy numbers elsewhere are about.

## Edge cases

**Edge case 1: the response has no references at all and the extraction is full of titles.** This
is the most common failure here. On the same 3,616 pilot responses, we measured 29.5% of
extracted units as absent from the response with the released prompts, falling to 6.9% with v2.
A real example: a Q&A page
with no references produced the extraction "Financial Accounting Theory; Intermediate Accounting;
…", which then failed the Semantic Scholar lookup and counted as hallucinated references.

Rule: if the response does not cite the title, the unit is `hallucinated` whether or not the paper
turns out to be real. Do not go looking for a way to credit the extraction.

**Edge case 2: the title is close but not exact.** The extractor often rewords, and our own
matching uses a loose rule for titles (80% of the words). If the response clearly refers to the
same work and the difference is wording, word order, or a dropped subtitle, treat it as cited →
judge on whether the paper is real. If the difference changes which paper it is, for example a
different year, different authors, or a different topic, treat it as not cited → `hallucinated`.
*Main round, 025* (confirmed at adjudication): "The Interior of Jupiter" is a real 2004
book chapter, but the response cites different authors and an ApJ article. A different work →
`hallucinated`.

**Edge case 3: a venue is not a work; a reference book is**. An extraction that is only a
journal or magazine name ("Nature", "Journal of Neurology") names where papers appear, not a paper
that was cited → `hallucinated`, even though the journal exists. *Main round*: 006, 014, 017, 038,
047. A reference book (an encyclopedia, a handbook) is itself a work, so judge it like any other
title. *Main round, 041*: "Encyclopedia of World Biography", found as an encyclopedia →
`supported`.

**Edge case 4: a generic title with nothing to pin it down**. The title is real, but
several works share it, and the response gives no author, year or venue that picks one out. You
cannot tell which work is cited, or whether any of them is → `unverifiable`. This is the
"ambiguous" case that "How to check existence" keeps `unverifiable` for. *Main round*: 001
("Health"), 026 ("Uterine Fibroids"), 044 (the response names an institute, but no work with that title
from it was found), and 009 (an author
and year are given, but no paper by them matches, and the title fits several works).

## Practical notes

- Two to three minutes per item, most of it searching.
- Read the response for the citation first. If it is not there, you are done, and you can skip the
  search.
- Work in sittings of about an hour.
