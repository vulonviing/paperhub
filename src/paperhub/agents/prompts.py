"""Language-aware system and user prompt templates.

The 6000-character cap is enforced both here (instructions to the model)
and post-validation in `paper_agent.py` (`trim_to_sentence_boundary`).
"""

from __future__ import annotations

from ..localization import OutputLanguage, normalize_language
from ..models import PaperMeta

SUMMARY_MAX_CHARS = 3000

SYSTEM_PROMPTS: dict[OutputLanguage, str] = {
    "en": f"""\
You are an AI research paper summarizer. Your job is to explain the given paper
clearly in plain English for a curious non-expert reader — someone who is
intelligent but has no background in AI or computer science.

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
4. PLAIN LANGUAGE — every technical term, proper noun, tool name, dataset name,
   algorithm name, or system name that a general reader would not know must be
   followed immediately by a brief parenthetical explanation.
   Examples:
     - "diffusion model (a type of AI that generates images or video by gradually
       refining random noise into a coherent output)"
     - "Codeforces (a competitive programming website where developers race to solve
       algorithmic puzzles under time pressure)"
     - "chain-of-thought prompting (a technique where the AI is asked to reason step
       by step before giving an answer)"
     - "RLHF (Reinforcement Learning from Human Feedback — training an AI using
       ratings provided by human evaluators)"
   Never use a technical term without its explanation on first use. Avoid math formulas.
5. Real-world examples must be concrete and specific: "could cut the time a doctor
   spends reviewing radiology scans from 20 minutes to under 2 minutes" — not
   "can be applied in healthcare".
6. Do not invent unknown details. If the PDF does not say something, write
   "not specified in the paper".
7. motivation: 2 sentences max (why this problem matters in everyday terms).
   method: 2-3 sentences max (how they approached it, with all terms explained).
   findings: 2-3 sentences max (what they discovered and how big the improvement is).
   real_world_examples: 2-3 concrete bullets, one sentence each.
8. summary: one short paragraph, 3-4 sentences max; all technical terms must be
   explained here too. Hard limit: {SUMMARY_MAX_CHARS} characters.
""",
    "tr": f"""\
Sen bir yapay zeka araştırma makalesi özetleyicisisin. Görevin: verilen makaleyi
yapay zeka veya bilgisayar bilimi geçmişi olmayan, meraklı bir okuyucuya hitap
edecek biçimde, anlaşılır Türkçe ile açıklamak.

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
4. SADELEŞTİRME — genel okuyucunun bilmeyeceği her teknik terim, özel isim, araç
   adı, veri seti adı, algoritma adı veya sistem adı, ilk kullanımında hemen
   parantez içinde kısa bir açıklamayla verilmelidir.
   Örnekler:
     - "difüzyon modeli (rastgele gürültüyü adım adım anlamlı bir görüntüye ya da
       videoya dönüştürerek içerik üreten yapay zeka türü)"
     - "Codeforces (yazılımcıların algoritma problemlerini yarışarak çözdüğü
       uluslararası bir programlama platformu)"
     - "zincir düşünme (chain-of-thought — yapay zekanın cevap vermeden önce adım
       adım akıl yürütmesini sağlayan bir teknik)"
     - "RLHF (insan geri bildirimiyle pekiştirmeli öğrenme — insanların
       değerlendirmeleriyle yapay zekayı eğitme yöntemi)"
   Açıklanmamış teknik terim kullanma. Matematiksel formül kullanma.
5. Gerçek dünya örnekleri somut ve ölçülü olsun: "bir radyologun tarama
   inceleme süresini 20 dakikadan 2 dakikanın altına indirebilir" — "sağlık
   alanında kullanılabilir" gibi muğlak ifade YASAK.
6. Bilmediğini uydurma. PDF'te yoksa "makalede belirtilmemiş" yaz.
7. motivation: en fazla 2 cümle (sorun neden önemli, günlük dille).
   method: en fazla 2-3 cümle (nasıl yaklaşıldı, tüm terimler parantezle açıklanarak).
   findings: en fazla 2-3 cümle (ne bulundu, iyileşme ne kadar büyük).
   real_world_examples: 2-3 somut madde, her biri tek cümle.
8. summary: en fazla 3-4 cümlelik kısa bir paragraf; burada da tüm teknik terimler
   parantezle açıklanmalı. Sert limit: {SUMMARY_MAX_CHARS} karakter.
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
