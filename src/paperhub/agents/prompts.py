"""Language-aware system and user prompt templates.

The 6000-character cap is enforced both here (instructions to the model)
and post-validation in `paper_agent.py` (`trim_to_sentence_boundary`).
"""

from __future__ import annotations

from ..localization import OutputLanguage, normalize_language
from ..models import PaperMeta

SUMMARY_MAX_CHARS = 6000

SYSTEM_PROMPTS: dict[OutputLanguage, str] = {
    "en": f"""\
You are an AI research paper summarizer. Your job is to summarize the given
paper clearly in English for a non-technical reader.

OUTPUT RULES (strict):
1. Return valid JSON only. No explanation, markdown fence, or preface.
2. Schema:
   {{
     "motivation": str,
     "method": str,
     "findings": str,
     "real_world_examples": [str, str, str],
     "summary": str
   }}
3. The summary field MUST NOT exceed {SUMMARY_MAX_CHARS} characters. This is a hard limit.
4. Translate jargon into plain language: explain "transformer" as a type of neural
   network that understands text; explain "RLHF" as training with human feedback;
   avoid mathematical formulas.
5. Make real-world examples concrete, such as "could reduce response time in a
   customer support chatbot"; vague claims like "can be used in many fields" are forbidden.
6. Do not invent unknown details. If the PDF does not say something, write
   "not specified in the paper".
7. motivation should be 2-4 sentences, method 3-6 sentences, findings 3-6
   sentences, and real_world_examples 2-4 concrete bullets.
8. summary should be consistent with motivation/method/findings/real_world_examples
   and read as one fluent paragraph or short English sections.
""",
    "tr": f"""\
Sen bir AI araştırma makalesi özetleyicisisin. Görevin: verilen makaleyi
TEKNİK OLMAYAN bir okuyucuya hitap edecek şekilde, Türkçe ve net biçimde özetlemek.

ÇIKTI KURALLARI (kesin):
1. Yalnızca geçerli JSON döndür. Açıklama, markdown fence, ön söz YOK.
2. Şema:
   {{
     "motivation": str,
     "method": str,
     "findings": str,
     "real_world_examples": [str, str, str],
     "summary": str
   }}
3. summary ALANI {SUMMARY_MAX_CHARS} KARAKTERİ AŞMASIN. Bu sert bir kısıttır.
4. Jargonu çevir: "transformer" → "metni anlayan sinir ağı türü";
   "RLHF" → "insan geri bildirimiyle eğitim"; matematiksel formül kullanma.
5. Gerçek dünya örnekleri için somut ol: "müşteri hizmetleri chatbot'unda yanıt
   süresini kısaltabilir" gibi; "çeşitli alanlarda kullanılabilir" gibi muğlak ifade YASAK.
6. Bilmediğini uydurma. PDF'te yoksa "makalede belirtilmemiş" yaz.
7. motivation 2-4 cümle, method 3-6 cümle, findings 3-6 cümle,
   real_world_examples 2-4 madde, hepsi somut ve net.
8. summary motivation/method/findings/real_world_examples ile uyumlu, akıcı bir
   tek paragraf veya kısa bölümlü Türkçe metin olsun.
""",
}

USER_TEMPLATES: dict[OutputLanguage, str] = {
    "en": """\
Paper title: {title}
Authors: {authors}
arXiv ID: {arxiv_id}
Abstract (HF):
{abstract}

Text extracted from the PDF (may be truncated):
\"\"\"
{pdf_text}
\"\"\"

Create the JSON summary according to the rules above. Return JSON only.
""",
    "tr": """\
Makale başlığı: {title}
Yazarlar: {authors}
arXiv ID: {arxiv_id}
Abstract (HF):
{abstract}

PDF'ten çıkarılan metin (kısaltılmış olabilir):
\"\"\"
{pdf_text}
\"\"\"

Yukarıdaki kurallara göre JSON özet üret. Yalnızca JSON döndür.
""",
}

FALLBACK_TEXT: dict[OutputLanguage, dict[str, str]] = {
    "en": {
        "abstract": "(no abstract found in the paper metadata)",
        "pdf_text": "(PDF text could not be extracted)",
        "authors": "(not specified)",
    },
    "tr": {
        "abstract": "(makalede abstract bulunamadı)",
        "pdf_text": "(PDF metni çıkarılamadı)",
        "authors": "(belirtilmemiş)",
    },
}

REPAIR_INSTRUCTIONS: dict[OutputLanguage, str] = {
    "en": (
        "Your previous response {issue}. Please regenerate the JSON output for "
        "the same paper; the summary field must be at most "
        f"{SUMMARY_MAX_CHARS}"
        " characters. Return JSON only."
    ),
    "tr": (
        "Önceki yanıtın {issue}. Lütfen aynı makale için JSON çıktısını yeniden üret; "
        "summary alanı en fazla "
        f"{SUMMARY_MAX_CHARS}"
        " karakter olsun. Yalnızca JSON döndür."
    ),
}


def build_messages(
    paper: PaperMeta,
    pdf_text: str,
    *,
    max_pdf_chars: int = 60_000,
    language: str | None = None,
) -> list[dict[str, str]]:
    """Build the user-message list passed to the LLM client."""

    lang = normalize_language(language)
    fallback = FALLBACK_TEXT[lang]
    abstract = paper.abstract or fallback["abstract"]
    truncated = pdf_text[:max_pdf_chars] if pdf_text else fallback["pdf_text"]
    user = USER_TEMPLATES[lang].format(
        title=paper.title,
        authors=", ".join(paper.authors) if paper.authors else fallback["authors"],
        arxiv_id=paper.arxiv_id,
        abstract=abstract,
        pdf_text=truncated,
    )
    return [{"role": "user", "content": user}]


def system_prompt(language: str | None = None) -> str:
    """Return the system prompt for a supported output language."""

    return SYSTEM_PROMPTS[normalize_language(language)]


def repair_instruction(language: str | None = None) -> str:
    """Return the repair prompt template for a supported output language."""

    return REPAIR_INSTRUCTIONS[normalize_language(language)]


SYSTEM_PROMPT = SYSTEM_PROMPTS["en"]
USER_TEMPLATE = USER_TEMPLATES["en"]
REPAIR_INSTRUCTION = REPAIR_INSTRUCTIONS["en"]
