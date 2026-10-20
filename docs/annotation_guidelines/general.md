# Annotation guideline: general rules for every task

Read this before your task's own guideline. It holds what adjudicating the 29 Biographies and
Historical Events disagreements showed. Unresolved references
("I", "you", "it", unnamed speakers) explained 9 of the 13 Biographies disagreements and 11 of the
16 Historical Events ones, so most of this file is about who a word refers to.

Where this file and a task guideline differ, this file wins: its rulings are newer. Every task
guideline now points here.

**Three rulings from adjudicating Summarization and Simplification in the main round:**
role nouns (rule 2), commentary about a text that is not there (rule 2) and
task talk (step 3 and rule 5). They were applied at adjudication;
the main-round annotators did not have them.

Examples marked with an item id are real pilot rows, already adjudicated. Examples marked
*main round* are main-round rows, added after all labelling and adjudication were
finished. Examples marked *illustrative* are written to show the rule.

## Which rules apply to which task

| Task | Unit | Rules |
|---|---|---|
| Biographies, Summarization, Simplification | one atom | the four-step test below, then rules 1, 2, 4 to 7 |
| Historical Events | the whole response | 3, 4, 7 |
| False Presuppositions, Rationalization (Numerical, Binary), Scientific Attribution | an extraction | none; you judge the extractor, not what the model claims |

## Start here: claim, not a claim, or unsure (Biographies, Summarization, Simplification)

**Why this section exists.** In the pilot, 9 of the 13 Biographies disagreements turned on
"I", "you", "It" or "A", and they all came from the same chain. The annotator was unsure whether
the atom was a claim, so, as the old rule said, labelled it as one. The article never mentions what
"I" or "It" did, and a silent source gives `hallucinated`. So "unsure" became `hallucinated`, while
the other annotator called the same atom `not a claim` (or read the word as the person). Both
followed the guideline as written; the guideline had no rule for these atoms. Here it is: steps 1
and 2 below decide them, so they never reach "unsure".

**First, what `unverifiable` means in Biographies, Summarization and Simplification.** It does
**not** mean "I could not verify it". It means "there is nothing here to verify". An atom the
source never mentions is `hallucinated`, not `unverifiable`, however hard you searched.
`unverifiable` always comes with a tag: `not a claim`, or `no article` when the source is missing.
If you are about to write `unverifiable` with `tag` blank, stop: you mean `hallucinated`.
(Historical Events is different: there `unverifiable` means you cannot tell what the response
commits to, and it takes no tag.)

**Then run every atom through four steps, in order. Stop at the first one that decides.**

| Step | Ask | If yes |
|---|---|---|
| 1 | Is the subject "I", "me", "we", "us" or "you"? | `not a claim`. Stop. (rule 1) |
| 2 | Is the subject "he", "she", "they", "it", "the speaker", a bare letter like "A", or a role noun like "the reviewer", **and** is there not exactly one person or thing in the row it could mean? (A role noun's only candidate is the passage's writer: rule 2.) | `not a claim`. Stop. (rule 2) |
| 3 | Is the atom a question, an instruction (including task talk: a requirement or rubric line about the task, even when written as a statement, rule 5), a framing or closing line, a fragment that states nothing ("The early years."), or a sentence that says nothing about its subject ("Gulshan Kumar is a fact.")? **Biographies only:** is it about something other than the person? | `not a claim`. Stop. (rule 5) |
| 4 | Is the wording broken, but you can still make out something it claims? | Label it as a claim by the source, leave `tag` blank, write the doubt in `notes`. This is the only place "unsure" applies (rule 6). Know where it leads: the source rarely supports such an atom, so it usually ends as `hallucinated`. |
| | None of the above | It is a claim. Judge it by the source: `supported` or `hallucinated`. |

**Things that are never reasons for `not a claim` or for "unsure":**

- the source does not mention it → `hallucinated`;
- you know it is true, but the source does not say it → `hallucinated`, with what you know in `notes`;
- it is vague ("He had a long career.") → a claim; vague atoms are judged like any other;
- it is an opinion or a reputation ("She is a renowned singer.") → a claim; see the evaluation edge
  case in your task guideline;
- it is about another topic than the source (Summarization, Simplification) → a claim,
  `hallucinated` (rule 5);
- the content fits, or does not fit, the person (rule 4).

**Worked through, from the pilot.** *Biographies 009*: the atom's subject is "I". Step 1 →
`not a claim`. No doubt to note. *Biographies 002*: the subject is "A", and nothing in the row
says who A is. Step 2 → `not a claim`. *Biographies 016*: the subject is "Mark Murphy" and it
says he is a songwriter. Steps 1 to 4 do not apply, so it is a claim; the article credits him with
song lyrics → `supported`.

## Rule 1 (Biographies, Summarization, Simplification): "I", "we" and "you" in an atom are the model or its reader

**Atom tasks only** (Biographies, Summarization, Simplification). Historical Events is different:
see rule 3.

In an atom task you see the atom, not the response it came from. "I", "me", "we", "us" and "you"
are the model talking or the reader being addressed. They never refer to a person in the source,
however well the rest of the atom fits that person (rule 4).

Such an atom asserts nothing the source could support or contradict → `unverifiable`, tag
`not a claim`.

- *Biographies 003*: "I" and "you" in the atom are the model and its reader, not the person.
- *Biographies 041*: an atom in which "I" works as a model, with Elsa Pataky's article open. Pataky
  is a model, but "I" is still the response speaking → `not a claim`.
- *Illustrative*, Summarization: "We now list the main points in order." → `not a claim`.

## Rule 2 (Biographies, Summarization, Simplification): "he", "she", "they", "it" and "the speaker" need exactly one candidate

A third-person word refers to a person (or thing) only when **exactly one** candidate in what you
are shown could be meant.

- **Biographies:** "He", "She", "They" and "The person" are the person named at the top
  (unchanged).
- **Summarization and Simplification:** the word refers to someone named in the atom itself, or to
  the one person the source is about. *Illustrative*: passage about one tennis player's comeback;
  atom "Her withdrawal would have been a mistake so early in her season." "Her" is the player →
  judge it as a claim (here an evaluation: Summarization edge case 3).
- **"It"** follows the same test. *Biographies 001 and 020*: "It" does not refer to the person
  and nothing else in the row fixes it → `not a claim`.

If no candidate, or more than one, could be meant, the atom has no subject you can check →
`unverifiable`, tag `not a claim`. *Illustrative*, Simplification: source sentence about the length
of a river; atom "The speaker wants to visit their family next year." No speaker exists in the
source → `not a claim`.

**Role nouns ("the author", "the narrator", "the reviewer")** (Summarization and Simplification). The word list above does not include role nouns, so they get their own test: **a role
noun counts as a subject only if it can be read as the passage's writer.** Then it is a claim about
the writer, judged by the passage. If it names a role nobody in the passage has, it has no
candidate → `not a claim`.

- Claim: *Summarization 020* (main round): "the author" of an opinion column with one author →
  judged by the passage, `hallucinated`. *Summarization 027* (main round): "the narrator", read as
  the article's writer → `hallucinated`.
- Not a claim: *Summarization 030* (main round): "the reviewer" in a news report that has no
  reviewer. *Summarization 042* (main round): "the French teacher", a role nobody in the passage
  has; a name elsewhere in the atom is not its subject.

**Commentary about a text that is not in the source**. An atom that comments on a
translation, a quote or a document the source does not contain has no subject you can check →
`not a claim`. *Simplification 048* (main round): "The English translation appears to be unrelated
to the Spanish quote…", where the source sentence holds no translation and no quote.

## Rule 3 (Historical Events): an unnamed speaker in a whole response

**Rule 1 does not apply here.** In the atom tasks you see one cut-out sentence, so "I" can only be
the model. In Historical Events you see the whole response, and base models often answer in
character, as a memoir or an interview. So "I" or "we" is one of three things:

| "I" or "we" is... | How you tell | Label |
|---|---|---|
| the model answering the question | it talks about the question or its knowledge: "I am not aware of any record of them meeting." | judge what it commits to; usually `supported` |
| one of the pair | the three conditions below all hold | `hallucinated` |
| a character you cannot place | any of the three conditions fails | `unverifiable` |

Read an unnamed "I", "he" or "we" as one of the pair only when all three hold:

1. the other person of the pair is named in the scene (a heading or title above the text is not
   the scene),
2. the unnamed speaker meets or deals with that person, and
3. no one else is offered as the speaker.

Then the response asserts the meeting → `hallucinated`.

- *045*: Grigol is named, and an unnamed "I" gave him a card and became his friend. No rival
  speaker → `hallucinated`.
- *049*: Cunard is named, and an unnamed "I" had lunch with him at the Sheraton in Manila in the
  1960s. No rival → `hallucinated`.

If any of the three fails, you cannot tell what the response commits to → `unverifiable`:

- a rival speaker: *019* (the "I" could be either of two men), *023* (the speaker made a film by
  another director), *026* ("he" is identified only as someone's father);
- neither of the pair named in the scene: *030*; *048* (the other of the pair is named only in a
  heading, "MANZOLI.", and the text calls him only "He");
- a group word that never resolves: *007* ("we"), *013* ("the two of them").

## Rule 4 (Biographies, Summarization, Simplification, Historical Events): fitting the person is never the test

Whether the content suits the person decides nothing, in either direction. *Biographies 041*: the
content fits Pataky, and "I" is still not Pataky. *Historical Events 045*: the speaker sounds
nothing like Shreve, and the response still asserts the meeting under rule 3. Decide who a word
refers to from what the text names, never from what would make sense.

## Rule 5 (Biographies, Summarization, Simplification): `not a claim` is narrow

`not a claim` (`unverifiable`) is only for an atom that asserts nothing a source could support or
contradict:

- framing and closing lines: "Here are the independent facts broken down from the passage:";
- questions and instructions: "What is her sign?", "Please paste the article you want
  summarised." (*illustrative*);
- task talk: a requirement or rubric line about the task. It counts as an instruction even
  when written as a statement, because step 3 is judged by what the atom does, not its grammar.
  *Main round*: *Summarization 022* ("There is a formatting requirement that needs to be
  checked."), *Summarization 049* ("Submissions will be evaluated for overall structure."),
  *Simplification 016* ("The paragraph of English text has no line exceeding 100 characters.").
  (Rule 2's pronoun list is read literally; step 3 is a category and is read by function. The
  two do not conflict.);
- the model or the reader as subject (rule 1);
- a fragment that states nothing: "The early years." (*illustrative*). A broken sentence in which
  you can still make out a claim is not a fragment: it is step 4, a claim with a note;
- a subject that does not resolve (rule 2).

**A factual statement about something else is still a claim.** In Summarization and
Simplification the question is whether the source entails the atom; a statement on another topic
is a claim the source does not support → `hallucinated`, tag blank. *Illustrative*: source
sentence about a 19th-century painter; atom "The app lets users export their data as a
spreadsheet." That is content the model made up → `hallucinated`.

**Biographies is the exception.** Its question asks about the person, so a statement about
something else ("The Digital Millennium Copyright Act was enacted in a year.") stays
`not a claim`, as in its edge case 1.

## Rule 6 (Biographies, Summarization, Simplification): "unsure" is for meaning, never for reference

"Unsure → claim, note the doubt" is step 4 of the test at the top of this file, and
nothing else. It covers a broken atom in which you can still make out a claim. (A fragment that
states nothing is step 3, `not a claim`.) It does not cover an atom whose
subject is "I", "you", an unresolved "it" or "the speaker": steps 1 and 2 settle those, with no
doubt left.

## Rule 7 (Biographies, Summarization, Simplification, Historical Events): other rulings from adjudication

- **(Biographies, Summarization, Simplification) Paraphrase counts.** *Biographies 016*: the article credits him with lyrics to songs; the atom
  says he is a songwriter → `supported`.
- **(Biographies) A fact about a relative is a claim about the person.** *Biographies 044*: the atom describes
  Xi Jinping's father, and the article covers it → judged as a claim, `supported`.
- **(Biographies, Summarization, Simplification) "Known for X" needs the reputation, not just X.** *Biographies 047*: the article says he wrote
  a handbook but names other work as what he is known for → `hallucinated`, tag `exaggerated`.
- **(Biographies, Summarization, Simplification) An unqualified generalization overstates.** *Biographies 030*: the atom says people disliked him,
  where the article gives him many enemies and many friends → `hallucinated`, tag `exaggerated`.
- **(Historical Events) A presupposed event is asserted.** *Historical Events 022*: the response refers to "the
  meeting" and places one of the pair as a visitor on the day the other arrived → `hallucinated`.
- **(Historical Events) A group that is never tied to the pair asserts nothing about them.** *Historical Events 047*:
  "the two men" is never identified as the pair, and the only scene narrated is with someone else
  → `supported`.

## Tags at adjudication (Biographies, Summarization, Simplification)

Annotators fill `tag` as before. At adjudication, the gold row's tag goes in the new `tag` column
of `adjudication.csv` beside `gold_label`: for the verifier's numbers, a row settled as
`not a claim` leaves the "without not a claim" split. The inter-annotator numbers still drop only
the rows both annotators tagged.
