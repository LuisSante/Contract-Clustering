"""Portuguese text views; technical origin and identifiers never become clustering features."""

from __future__ import annotations

import re
import unicodedata

BANKS = {
    "bmg": [r"\bbmg\b", r"banco bmg"],
    "pan": [r"\bpan\b"],
    "banco_do_brasil": [
        r"banco\s+do\s+brasil",
        r"\bbb\b",
        r"ourocard",
        r"poupan[cç]a ouro",
        r"poupex",
    ],
    "cencosud_bradescard": [r"cencosud", r"bradescard"],
    "cetelem": [r"cetelem", r"bnp paribas"],
    "itau": [r"ita[uú]", r"itaucard", r"hipercard"],
    "bradesco": [r"bradesco"],
    "santander": [r"santander", r"ol[eé] consignado"],
    "facta": [r"facta"],
    "capital_consig": [r"capital consig"],
    "daycoval": [r"daycoval"],
    "mercantil": [r"mercantil do brasil", r"banco mercantil"],
    "master": [r"banco master", r"credcesta", r"credicesta"],
    "agibank": [r"agibank"],
    "credsystem": [r"cred[ -]?system", r"esposende"],
    "nubank": [r"nubank", r"nu pagamentos"],
    "caixa": [r"caixa econ[oô]mica"],
    "safra": [r"banco safra"],
    "digio": [r"digio"],
    "c6": [r"banco c6", r"c6 bank", r"c6 consignado"],
    "inter": [r"banco inter\b"],
    "banrisul": [r"banrisul"],
}
BRANDS = [
    r"le biscuit",
    r"riachuelo",
    r"renner",
    r"pernambucanas",
    r"carrefour",
    r"cea\b",
    r"c\s*&\s*a",
    r"bandeira (?:visa|mastercard)",
]
STOPWORDS = [
    "a",
    "ao",
    "aos",
    "aquela",
    "aquelas",
    "aquele",
    "aqueles",
    "aqui",
    "as",
    "ate",
    "com",
    "como",
    "da",
    "das",
    "de",
    "dela",
    "dele",
    "deles",
    "depois",
    "do",
    "dos",
    "e",
    "ela",
    "elas",
    "ele",
    "eles",
    "em",
    "entre",
    "era",
    "eram",
    "essa",
    "essas",
    "esse",
    "esses",
    "esta",
    "estas",
    "este",
    "estes",
    "eu",
    "foi",
    "foram",
    "ha",
    "isso",
    "isto",
    "ja",
    "lhe",
    "lhes",
    "mais",
    "mas",
    "me",
    "mesmo",
    "meu",
    "meus",
    "minha",
    "minhas",
    "muito",
    "na",
    "nas",
    "nao",
    "nem",
    "no",
    "nos",
    "nossa",
    "nossas",
    "nosso",
    "nossos",
    "numa",
    "num",
    "nunca",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "qual",
    "quando",
    "que",
    "quem",
    "se",
    "sem",
    "seu",
    "seus",
    "sua",
    "suas",
    "sao",
    "ser",
    "sera",
    "sobre",
    "so",
    "sua",
    "suas",
    "tambem",
    "tem",
    "tendo",
    "tenha",
    "tenho",
    "ter",
    "toda",
    "todas",
    "todo",
    "todos",
    "um",
    "uma",
    "umas",
    "uns",
    "voce",
    "voces",
    "vos",
    "cujo",
    "cuja",
    "cujos",
    "cujas",
    "desta",
    "deste",
    "destes",
    "destas",
    "deste",
    "desse",
    "dessa",
    "desses",
    "dessas",
    "mediante",
    "assim",
    "bem",
    "conforme",
    "atraves",
    "respectivo",
    "respectivos",
    "respectiva",
    "respectivas",
    "referido",
    "referida",
    "referidos",
    "referidas",
    "onde",
    "cada",
    "demais",
    "caso",
    "forma",
    "presente",
    "termos",
    "disposto",
    "artigo",
    "paragrafo",
    "inciso",
    "seguinte",
    "seguintes",
    "previsto",
    "prevista",
    "previstas",
    "previstos",
    "inclusive",
    "dentre",
    "sendo",
    "seja",
    "sejam",
    "devera",
    "podera",
]


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def institutions(text: str) -> list[str]:
    return [
        name
        for name, patterns in BANKS.items()
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    ]


def clean(text: str, *, mask_institutions: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n(?=\w)", "", text)
    text = re.sub(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}", " ", text)
    text = re.sub(r"\b\d{2,3}(?:\.\d{3}){2}[/-]\d{2,4}(?:-\d{2})?\b", " ", text)
    text = re.sub(r"(?:https?://|www\.)\S+|\b\S+@\S+\.\S+", " ", text)
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", text)
    text = re.sub(r"\d+(?:[.,]\d+)?\s*%", " percentual ", text)
    text = re.sub(r"\b\d{4,}\b", " ", text)
    text = re.sub(r"\[?\s*[…]+\s*\]?|\.{3,}|[_☐☑]+", " ", text)
    text = re.sub(
        r"(?im)^.*(?:assinado eletronicamente|c[oó]digo de valida[cç][aã]o|projudi).*$", " ", text
    )
    if mask_institutions:
        for patterns in BANKS.values():
            for pattern in patterns:
                text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        for pattern in BRANDS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return canonical(text)


def segments(page_text: str, *, max_words: int = 180, overlap: int = 25) -> list[dict]:
    """Heuristic clause/window units; offsets refer to the processed PDF's page text."""
    if max_words <= overlap or overlap < 0:
        raise ValueError("max_words must exceed nonnegative overlap")
    breaks = (
        [0]
        + [
            m.start()
            for m in re.finditer(
                r"(?m)^\s*(?:CL[ÁA]USULA\s+\d+|\d{1,2}(?:\.\d{1,2}){0,2}[.)-]\s+|[IVX]{1,5}\s*[-–])",
                page_text,
            )
        ]
        + [len(page_text)]
    )
    result = []
    for first, last in zip(sorted(set(breaks)), sorted(set(breaks))[1:]):
        words = list(re.finditer(r"\S+", page_text[first:last]))
        for i in range(0, len(words), max_words - overlap):
            selected = words[i : i + max_words]
            if len(selected) < 12:
                continue
            start, end = first + selected[0].start(), first + selected[-1].end()
            value = clean(page_text[start:end])
            if len(value) >= 80:
                result.append({"start": start, "end": end, "text": value})
            if i + max_words >= len(words):
                break
    return result
