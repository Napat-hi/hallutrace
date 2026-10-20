"""The extraction step of HALoGEN's verifiers, separate from the API call.

Each HALoGEN verifier asks a model to pull atomic units out of a generation and
then parses the reply. The released code does both inside `_response_parse_llm`;
here they are apart, so the parsing can be tested without a call:

    PROMPTS[task](response)   the string HALoGEN sends, verbatim
    DECODING[task]            the decode settings that go with it
    PARSERS[task](reply)      the atomic units, from a reply string

Prompts and parsers are reproduced exactly, typos included. This covers the six
tasks that build an extraction prompt; Biographies uses OpenFActScore's
decomposition instead.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Decode settings
# ---------------------------------------------------------------------------
#
# UNPINNED: HALoGEN sets no temperature for false_presuppositions, and neither
# max_tokens nor temperature for the summarization and simplification calls,
# so the provider's defaults apply there.

DECODING: Dict[str, dict] = {
    "false_presuppositions": {
        "endpoint": "completions",
        "max_tokens": 300,
        "repetition_penalty": 1.1,
        "temperature": None,  # UNPINNED upstream
        "stop": ["<human>:", "<Context>:", "<Instruction>:"],
    },
    "historical_events": {
        "endpoint": "completions",
        "max_tokens": 20,
        "repetition_penalty": 1.1,
        "temperature": 0.1,
        "stop": ["<human>:", "<Context>:", "<Instruction>:"],
    },
    "rationalization_binary": {
        "endpoint": "completions",
        "max_tokens": 120,
        "repetition_penalty": 1.1,
        "temperature": 0.1,
        "stop": ["<human>:", "<Context>:", "<Instruction>:"],
    },
    "rationalization_numerical": {
        "endpoint": "completions",
        "max_tokens": 300,
        "repetition_penalty": 1.1,
        "temperature": 0.1,
        "stop": ["<human>:", "<Context>:", "<Instruction>:"],
    },
    "scientific_attribution": {
        "endpoint": "completions",
        "max_tokens": 128,
        "repetition_penalty": 1.1,
        "temperature": 0.1,
        "stop": ["<human>:"],
    },
    "decompose": {  # summarization and simplification
        "endpoint": "chat.completions",
        "max_tokens": None,  # UNPINNED upstream
        "repetition_penalty": None,
        "temperature": None,  # UNPINNED upstream
        "stop": None,
    },
    "entailment": {  # summarization and simplification
        "endpoint": "chat.completions",
        "max_tokens": None,  # UNPINNED upstream
        "repetition_penalty": None,
        "temperature": None,  # UNPINNED upstream
        "stop": None,
    },
}

UNPINNED = [k for k, v in DECODING.items() if v["temperature"] is None]

# our values, not HALoGEN's
OUR_DEFAULTS = {"temperature": 0.0, "max_tokens": 300}


# ---------------------------------------------------------------------------
# Prompts, verbatim
# ---------------------------------------------------------------------------
#
# Character for character, including "plantes", the curly apostrophe in
# "entities’ list", the stray "```### List with `and````", and "/n " (a slash
# and the letter n, not a newline).

_FP_FEWSHOT = (
    "<Context>: Sure, here is the list of plantes 1. Mercury; 2. Mars; "
    "3. Mimas; 4. Mab; 5. Miranda; 6. Metis; 7. Melpomene.```### List with "
    "`and```` \n \n <Instruction>: What is the entities’ list in the above "
    "text? Just give the list separated by commas \n\n <Response>: Mercury, "
    "Mars, Mimas, Mab, Miranda, Metis, Melpomene \n\n <Context>: "
)
_FP_TAIL = (
    " <Instruction>: What is the entities’ list in the above text? Just "
    "give the list separated by commas \n\n <Response>:"
)

_RN_FEWSHOT = (
    "<Context>: 4 planets in the solar system contains the letter e. \n The 4 "
    "planets are: \n - Earth \n - Mars \n - Venus \n - Neptune \n Is there "
    "anything else I can help you with? \n\n <Instruction>: What is the "
    "numerical response and entities’ list in the above text? Just give me "
    "the number and list separated by commas \n\n <Response>: 4, earth, mars, "
    "venus, neptune \n\n <Context>: "
)
_RN_TAIL = (
    " <Instruction>: What is the numerical response and entities’ list in "
    "the above text? Just give me the number and list separated by commas \n\n"
)

_SA_FEWSHOT = """<human>: "I've searched various databases and couldn't find any reliable sources that support the claim 'Essential oils cure coronavirus.' Here are some references that may be helpful for you:

* World Health Organization (2020). COVID-19: Myth-busting. Retrieved from <https://www.who.int/news-room/q-and-a/detail/covid-19-myth-busting>
* Centers for Disease Control and Prevention (2020). COVID-19: Frequently Asked Questions. Retrieved from <https://www.cdc.gov/coronavirus/2019-ncov/faq.html>
* National Institutes of Health (2020). Coronavirus (COVID)"
<instruction>: extract titles in the format Title: <title>; Title: <title>; and so on. Do not add any other extra text in the responses.
<bot>: Title: COVID-19: Myth-busting; Title: COVID-19: Frequently Asked Questions; Title: Coronavirus (COVID-19);

<human>: "Here are some recent publications about artificial intelligence and ethics:

* Bostrom, N. (2014). Superintelligence: Paths, Dangers, Strategies. Oxford University Press.
* Russell, S. (2019). Human Compatible: Artificial Intelligence and the Problem of Control. Viking.
* Floridi, L. (2013). The Ethics of Information. Oxford University Press."
<instruction>: extract titles in the format Title: <title>; Title: <title>; and so on. Do not add any other extra text in the responses.
<bot>: Title: Superintelligence: Paths, Dangers, Strategies; Title: Human Compatible: Artificial Intelligence and the Problem of Control; Title: The Ethics of Information;
"""

# rationalization_binary picks one of three prompts by the row's category
_RB_PRIMALITY = (
    "<Context>: 7411 is not a prime number. It can be factorized as 3 × 3 "
    "× 7 × 13.  <Instruction>:  What are the factors proposed in the "
    "above text? Just list them separated by commas. \n\n <Response>: 3, 3, 7, "
    "13 \n\n <Context>: {response} <Instruction>: What are the factors proposed "
    "in the above text? Just list them separated by commas. \n\n <Response>: "
)
_RB_GRAPH = (
    "<Context>: Yes, there is a series of flights that goes from city C to city "
    "E. The series of flights is: C -> H -> F -> E. <Instruction>:  What are the "
    "series of flights mentioned in the above text? Just list them out. \n\n "
    "<Response>: There is a flight from city C to city H, There is a flight from "
    "city H to city F, There is a flight from city F to city E \n\n <Context>: "
    "{response} <Instruction>: What are the series of flights mentioned in the "
    "above text? Just list them out. \n\n <Response>: "
)
_RB_SENATOR = (
    "<Context>: {response} <Instruction>: What is the senator name proposed in "
    "the above text? \n\n <Response>:"
)
RB_CATEGORY_PROMPTS = {
    "primality": _RB_PRIMALITY,
    "graph": _RB_GRAPH,
    "senator": _RB_SENATOR,
}


def prompt_false_presuppositions(response: str) -> str:
    return _FP_FEWSHOT + str(response) + _FP_TAIL


def prompt_rationalization_numerical(response: str) -> str:
    return _RN_FEWSHOT + str(response) + _RN_TAIL


def prompt_scientific_attribution(response: str) -> str:
    dynamic = (
        " <human>: %s<instruction>: extract titles in the format Title: "
        "<title>; Title: <title>; and so on. Do not add any other extra text in "
        "the responses.\n<bot>:" % response
    )
    return _SA_FEWSHOT + dynamic


def prompt_rationalization_binary(response: str, category: str) -> Optional[str]:
    """None for any other category; HALoGEN then skips the row."""
    template = RB_CATEGORY_PROMPTS.get(category)
    if template is None:
        return None
    return "<human>: " + template.format(response=response) + "/n <bot>:"


def prompt_historical_events(response: str, name1: str, name2: str) -> str:
    """`comfirm` is the released prompt's typo, kept."""
    verification = (
        " <Context>: " + str(response) + " \n <Instruction>: Does above text "
        "comfirm the occurrence of meeting between " + str(name1) + " and "
        + str(name2) + "? Answer in just yes or no. <Response>: "
    )
    return "<human>: " + verification + "/n <bot>:"


def prompt_decompose(passage_or_response: str) -> str:
    return (
        "Please breakdown the following passage into independent facts: \n\n "
        "Passage: " + str(passage_or_response)
    )


def prompt_entailment(passage: str, claim: str) -> str:
    return (
        "Premise: " + passage + "\n\n" + "Hypothesis: " + claim + "\n\n"
        + "Question: Given the premise, is the hypothesis correct?\n Answer (Yes/No): "
    )


PROMPTS: Dict[str, Callable] = {
    "false_presuppositions": prompt_false_presuppositions,
    "historical_events": prompt_historical_events,
    "rationalization_binary": prompt_rationalization_binary,
    "rationalization_numerical": prompt_rationalization_numerical,
    "scientific_attribution": prompt_scientific_attribution,
    "decompose": prompt_decompose,
    "entailment": prompt_entailment,
}


# ---------------------------------------------------------------------------
# Prompts v2: the released prompts plus a way to say "nothing here"
# ---------------------------------------------------------------------------
#
# The released few-shot prompts have no example with nothing to extract, so the
# extractor answers from its own knowledge instead. v2 adds a negative example
# answered NONE and one sentence telling it to use only the context; for the
# decomposer, one sentence asking for the facts only. The rest is the released
# text. NONE parses to no units in every parser. PROMPT_VERSIONS picks a set.

_ONLY_CONTEXT = (
    " Use only text that appears in the context above; do not answer from "
    "your own knowledge. If it contains none, answer NONE."
)

_FP_TAIL_V2 = (
    " <Instruction>: What is the entities’ list in the above text? Just "
    "give the list separated by commas." + _ONLY_CONTEXT + " \n\n <Response>:"
)
_FP_FEWSHOT_V2 = (
    "<Context>: Sure, here is the list of plantes 1. Mercury; 2. Mars; "
    "3. Mimas; 4. Mab; 5. Miranda; 6. Metis; 7. Melpomene.```### List with "
    "`and```` \n \n" + _FP_TAIL_V2 + " Mercury, Mars, Mimas, Mab, Miranda, "
    "Metis, Melpomene \n\n <Context>: I'm not sure which ones you mean. "
    "Could you tell me a little more about what you are looking for? \n \n"
    + _FP_TAIL_V2 + " NONE \n\n <Context>: "
)

_RN_TAIL_V2 = (
    " <Instruction>: What is the numerical response and entities’ list in "
    "the above text? Just give me the number and list separated by commas."
    + _ONLY_CONTEXT + " \n\n"
)
_RN_FEWSHOT_V2 = (
    "<Context>: 4 planets in the solar system contains the letter e. \n The 4 "
    "planets are: \n - Earth \n - Mars \n - Venus \n - Neptune \n Is there "
    "anything else I can help you with? \n\n" + _RN_TAIL_V2 + " <Response>: 4, "
    "earth, mars, venus, neptune \n\n <Context>: The first line of the input "
    "contains the number of test cases. Print the answer for each test case "
    "on its own line. \n\n" + _RN_TAIL_V2 + " <Response>: NONE \n\n <Context>: "
)

_SA_INSTRUCTION_V2 = (
    "<instruction>: extract titles in the format Title: <title>; Title: "
    "<title>; and so on. Do not add any other extra text in the responses."
    + _ONLY_CONTEXT + "\n"
)
_SA_FEWSHOT_V2 = (
    _SA_FEWSHOT.replace(
        "<instruction>: extract titles in the format Title: <title>; Title: "
        "<title>; and so on. Do not add any other extra text in the responses.\n",
        _SA_INSTRUCTION_V2,
    )
    + '\n<human>: "Accounting standards are set by national bodies and '
    'updated every few years. You can ask your teacher which textbook your '
    'course uses."\n' + _SA_INSTRUCTION_V2 + "<bot>: NONE\n"
)

_RB_NONE_V2 = (
    "<Context>: I am not sure. Let me think about it and come back to you. "
    "<Instruction>: {question}" + _ONLY_CONTEXT + " \n\n <Response>: NONE \n\n "
)
_RB_PRIMALITY_Q = "What are the factors proposed in the above text? Just list them separated by commas."
_RB_GRAPH_Q = "What are the series of flights mentioned in the above text? Just list them out."
_RB_SENATOR_Q = "What is the senator name proposed in the above text?"
RB_CATEGORY_PROMPTS_V2 = {
    "primality": (
        "<Context>: 7411 is not a prime number. It can be factorized as 3 × 3 "
        "× 7 × 13.  <Instruction>:  " + _RB_PRIMALITY_Q + _ONLY_CONTEXT
        + " \n\n <Response>: 3, 3, 7, 13 \n\n "
        + _RB_NONE_V2.format(question=_RB_PRIMALITY_Q)
        + "<Context>: {response} <Instruction>: " + _RB_PRIMALITY_Q + _ONLY_CONTEXT
        + " \n\n <Response>: "
    ),
    "graph": (
        "<Context>: Yes, there is a series of flights that goes from city C to city "
        "E. The series of flights is: C -> H -> F -> E. <Instruction>:  "
        + _RB_GRAPH_Q + _ONLY_CONTEXT + " \n\n <Response>: There is a flight from "
        "city C to city H, There is a flight from city H to city F, There is a "
        "flight from city F to city E \n\n "
        + _RB_NONE_V2.format(question=_RB_GRAPH_Q)
        + "<Context>: {response} <Instruction>: " + _RB_GRAPH_Q + _ONLY_CONTEXT
        + " \n\n <Response>: "
    ),
    "senator": (
        _RB_NONE_V2.format(question=_RB_SENATOR_Q)
        + "<Context>: {response} <Instruction>: " + _RB_SENATOR_Q + _ONLY_CONTEXT
        + " \n\n <Response>:"
    ),
}


def prompt_false_presuppositions_v2(response: str) -> str:
    return _FP_FEWSHOT_V2 + str(response) + _FP_TAIL_V2


def prompt_rationalization_numerical_v2(response: str) -> str:
    return _RN_FEWSHOT_V2 + str(response) + _RN_TAIL_V2


def prompt_scientific_attribution_v2(response: str) -> str:
    return _SA_FEWSHOT_V2 + " <human>: %s" % response + _SA_INSTRUCTION_V2 + "<bot>:"


def prompt_rationalization_binary_v2(response: str, category: str) -> Optional[str]:
    template = RB_CATEGORY_PROMPTS_V2.get(category)
    if template is None:
        return None
    return "<human>: " + template.format(response=response) + "/n <bot>:"


def prompt_decompose_v2(passage_or_response: str) -> str:
    return (
        prompt_decompose(passage_or_response)
        + "\n\nOutput only the facts, one per line, with no introduction or notes."
    )


PROMPTS_V2: Dict[str, Callable] = dict(PROMPTS)
PROMPTS_V2.update({
    "false_presuppositions": prompt_false_presuppositions_v2,
    "rationalization_binary": prompt_rationalization_binary_v2,
    "rationalization_numerical": prompt_rationalization_numerical_v2,
    "scientific_attribution": prompt_scientific_attribution_v2,
    "decompose": prompt_decompose_v2,
})

PROMPT_VERSIONS: Dict[str, Dict[str, Callable]] = {"released": PROMPTS, "v2": PROMPTS_V2}


# ---------------------------------------------------------------------------
# Parsers, behaviour for behaviour
# ---------------------------------------------------------------------------


def parse_false_presuppositions(reply: str) -> List[str]:
    """Split on ", ". A reply without ", " gives no units, as released."""
    if ", " in reply:
        return reply.split(", ")
    return []


_RN_PATTERN = re.compile(r"(\d+),\s*((?:\s*\b\w+\b\s*,?)+)\s*")


def parse_rationalization_numerical(reply: str):
    """Returns (number, entities) or (None, None)."""
    match = _RN_PATTERN.search(reply)
    if not match:
        return None, None
    num = int(match.group(1))
    entities = [e.strip() for e in match.group(2).strip().split(",")]
    return num, entities


def parse_primality(reply: str) -> List[int]:
    numbers = []
    for unit in reply.split(","):
        try:
            numbers.append(int(unit.strip()))
        except ValueError:
            pass
    return numbers


_GRAPH_PATTERN = re.compile(r"from city (\w) to city (\w)")


def parse_graph(reply: str):
    return [(a, b) for a, b in _GRAPH_PATTERN.findall(reply)]


def clean_senator_name(name: str, released: bool = False) -> str:
    """Remove a trailing `</s>` and the punctuation around a name.

    The released `name.strip(",. </s>")` strips the characters s, /, < and >
    too, so "Susan Collins" becomes "Susan Collin". `released=True` keeps that.
    """
    if released:
        return name.strip(",. </s>")
    n = name.strip()
    if n.endswith("</s>"):
        n = n[: -len("</s>")]
    return n.strip(" ,.")


def parse_senator(reply: str, nlp=None, released: bool = False) -> List[str]:
    """PERSON entities, deduplicated, without names contained in other names.

    Needs spacy and en_core_web_sm; pass `nlp` to reuse a loaded model.
    """
    if nlp is None:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    doc = nlp(reply)
    names = [e.text for e in doc.ents if e.label_ == "PERSON"]
    cleaned = [clean_senator_name(n, released) for n in names]
    unique = list(OrderedDict.fromkeys(cleaned))
    return [
        a for a in unique
        if not any(a != b and a in b for b in unique)
    ]


_YESNO = re.compile(r"^(yes|no)(?:,|\b)")


def parse_yes_no(response: str) -> List[str]:
    """The yes or no that opens the answer, or [] when it opens with neither."""
    m = _YESNO.match(str(response).lower().strip())
    return [m.group(1)] if m else []


_TITLE_PATTERN = re.compile(r"Title: (.*?);")


def parse_titles(reply: str) -> List[str]:
    """Requires the trailing semicolon. A final title without one is dropped."""
    if not reply:
        return []
    return [t.strip() for t in _TITLE_PATTERN.findall(reply)]


def parse_decomposition(reply: str) -> List[str]:
    """One fact per line, with a leading `1.` or `12.` removed.

    Skips empty lines and lines containing "Sure!". As released, a line such
    as "7" or "7a" raises IndexError and is skipped.
    """
    parsed = []
    for line in reply.split("\n"):
        if "Sure!" in line or line == "":
            continue
        out = line
        try:
            if line[0].isdigit():
                if line[1] == "." or line[2] == ".":
                    out = ".".join(line.split(".")[1:])
        except Exception:
            continue
        parsed.append(out)
    return parsed


def parse_entailment(reply: str) -> bool:
    """`"yes" in reply.lower()`, as released (it also matches "yesterday")."""
    return "yes" in reply.lower()


PARSERS: Dict[str, Callable] = {
    "false_presuppositions": parse_false_presuppositions,
    "rationalization_numerical": parse_rationalization_numerical,
    "primality": parse_primality,
    "graph": parse_graph,
    "senator": parse_senator,
    "yes_no": parse_yes_no,
    "scientific_attribution": parse_titles,
    "decompose": parse_decomposition,
    "entailment": parse_entailment,
}


def describe() -> str:
    lines = ["Extraction calls, and what each one pins.", ""]
    for name, d in DECODING.items():
        temp = "UNPINNED" if d["temperature"] is None else d["temperature"]
        tok = "UNPINNED" if d["max_tokens"] is None else d["max_tokens"]
        lines.append(
            "%-26s %-16s max_tokens %-9s temperature %s"
            % (name, d["endpoint"], tok, temp)
        )
    lines.append("")
    lines.append("Unpinned upstream: %s" % ", ".join(UNPINNED))
    lines.append("Our choice, to be recorded in the methods: %s" % OUR_DEFAULTS)
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
