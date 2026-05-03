"""Language-aware system and user prompt templates.

Cloud prompts ask for a 3000-character summary; Ollama prompts use a shorter
1000-character target for local generation speed. `PaperSummary` keeps a
6000-character model-level guard for backward compatibility.
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

# ---------------------------------------------------------------------------
# Ollama-specific user templates
#
# Small local models (gemma, llama, etc.) tend to collapse all content into the
# "summary" field and leave the other fields empty when given open-ended
# instructions. These templates use a "fill-in-each-field" style — explicitly
# asking a question per field — which small models handle much more reliably
# because it reads like a structured cloze task rather than free generation.
# ---------------------------------------------------------------------------

OLLAMA_USER_TEMPLATES: dict[OutputLanguage, str] = {
    "en": """\
Paper title: {title}
Authors: {authors}
arXiv ID: {arxiv_id}
Abstract:
{abstract}

PDF text (may be truncated):
\"\"\"
{pdf_text}
\"\"\"

Read the paper above, then answer each of the following questions to build a JSON object.
Every field below MUST contain real text from the paper — do not leave any field empty.

Answer these questions for each JSON field:
- "motivation": Why does this research problem matter in everyday life? Write 2 sentences.
- "method": How did the researchers approach and solve the problem? Write 2-3 sentences.
- "findings": What did they discover and how significant is the improvement? Write 2-3 sentences.
- "real_world_examples": Give 2-3 concrete, specific real-world applications with numbers or metrics.
- "summary": Write a plain-language 3-4 sentence paragraph summarizing the paper (max {max_chars} characters).

Use plain language throughout. Explain every technical term in parentheses on first use.
Return ONLY the JSON object — no preamble, no markdown.
""",
    "tr": """\
Makale başlığı: {title}
Yazarlar: {authors}
arXiv ID: {arxiv_id}
Özet:
{abstract}

PDF metni (kısaltılmış olabilir):
\"\"\"
{pdf_text}
\"\"\"

Yukarıdaki makaleyi oku, ardından aşağıdaki her soruyu yanıtlayarak bir JSON nesnesi oluştur.
Aşağıdaki her alan makaleden gerçek metin içermelidir — hiçbir alanı boş bırakma.

Her JSON alanı için şu soruları yanıtla:
- "motivation": Bu araştırma problemi günlük hayatta neden önemlidir? 2 cümle yaz.
- "method": Araştırmacılar problemi nasıl ele aldı ve çözdü? 2-3 cümle yaz.
- "findings": Ne keşfettiler ve iyileşme ne kadar büyük? 2-3 cümle yaz.
- "real_world_examples": Sayı veya ölçüm içeren 2-3 somut, özgün gerçek dünya uygulaması ver.
- "summary": Makaleyi özetleyen sade dilde 3-4 cümlelik bir paragraf yaz (maks {max_chars} karakter).

Her yerde sade dil kullan. Her teknik terimi ilk kullanımında parantez içinde açıkla.
YALNIZCA JSON nesnesini döndür — önsöz veya markdown olmadan.
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
    provider: str | None = None,
) -> list[dict[str, str]]:
    """Build the user-message list passed to the LLM client."""

    lang = normalize_language(language)
    fallback = FALLBACK_TEXT[lang]
    abstract = paper.abstract or fallback["abstract"]
    truncated = pdf_text[:max_pdf_chars] if pdf_text else fallback["pdf_text"]
    fmt_args = {
        "title": paper.title,
        "authors": ", ".join(paper.authors) if paper.authors else fallback["authors"],
        "arxiv_id": paper.arxiv_id,
        "abstract": abstract,
        "pdf_text": truncated,
    }
    if (provider or "").lower() == "ollama":
        # Always use English template for Ollama (fewer tokens, clearer instructions).
        # Append language note when output is not English.
        user = OLLAMA_USER_TEMPLATES["en"].format(max_chars=OLLAMA_SUMMARY_MAX_CHARS, **fmt_args)
        note = _OLLAMA_LANG_NOTES.get(lang, "")
        if note:
            user = user.rstrip() + f"\n\n{note}\n"
    else:
        user = USER_TEMPLATES[lang].format(**fmt_args)
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

# ---------------------------------------------------------------------------
# Ollama-specific prompts
#
# Small local models (gemma, llama, etc.) are less reliable at following the
# elaborate multi-rule prompts above. These prompts are shorter and more direct.
#
# LANGUAGE STRATEGY FOR OLLAMA
# Always use English instructions regardless of the desired output language.
# Reason: Turkish (and other non-English) text tokenises into ~2× more tokens
# than English with typical LLM tokenisers, because those tokenisers are
# primarily trained on English. The effect compounds:
#   • Turkish system + user prompts  → ~2× more input tokens to process
#   • Turkish JSON output (summary alone at 3000 chars) → ~1500 output tokens
#     vs ~750 for English — directly doubling generation time per paper
# For gemma4:e4b at ~10 tok/s this means 5–9 min per paper in Turkish mode
# vs 2–3 min in English mode.
# Solution: use the English prompt always; append a one-line language note
# ("Write all field values in Turkish") so the output language is still correct
# but the heavy instruction-following load is done entirely in English.
# ---------------------------------------------------------------------------

# Reduced summary cap for Ollama — keeps output token count manageable.
# Cloud providers can handle 3000 chars fine; local models slow down sharply
# beyond ~1000 chars of non-English output.
OLLAMA_SUMMARY_MAX_CHARS = 1000

# One-line instruction appended to English prompts when output lang != English.
_OLLAMA_LANG_NOTES: dict[str, str] = {
    "tr": (
        "IMPORTANT: Write all content inside the JSON field values in Turkish (Türkçe). "
        'Keep the JSON keys exactly as-is: "motivation", "method", "findings", '
        '"real_world_examples", "summary".'
    ),
}

OLLAMA_SYSTEM_PROMPT_EN = f"""\
You are a JSON generator. Output ONLY a valid JSON object — no text before or after it, no markdown, no explanation.

You MUST include all five keys in your JSON output:

{{
  "motivation": "<why this research problem matters in everyday terms — 2 sentences max>",
  "method": "<how the researchers approached the problem — 2-3 sentences max>",
  "findings": "<what they discovered and how significant the improvement is — 2-3 sentences max>",
  "real_world_examples": ["<concrete example 1>", "<concrete example 2>"],
  "summary": "<short paragraph of 3-4 sentences — hard limit: {OLLAMA_SUMMARY_MAX_CHARS} characters>"
}}

Rules:
- Explain every technical term in parentheses on first use, e.g. "transformer (an AI model that processes text in parallel)"
- real_world_examples must be specific: "reduces doctor review time from 20 min to 2 min" not "useful in healthcare"
- summary must not exceed {OLLAMA_SUMMARY_MAX_CHARS} characters
- If something is not stated in the paper, write "not specified in the paper"
- Output JSON ONLY
"""


def system_prompt_for_provider(
    language: str | None = None,
    provider: str | None = None,
) -> str:
    """Return the system prompt appropriate for the given provider and language.

    OpenAI, Anthropic, and Google receive the full elaborated prompt in the
    requested language.  Ollama always receives the compact English prompt
    (faster tokenisation, better instruction-following on small models), with
    a language note appended when the output language is not English.
    """
    lang = normalize_language(language)
    if (provider or "").lower() == "ollama":
        prompt = OLLAMA_SYSTEM_PROMPT_EN
        note = _OLLAMA_LANG_NOTES.get(lang, "")
        if note:
            prompt = prompt.rstrip() + f"\n\n{note}\n"
        return prompt
    return SYSTEM_PROMPTS[lang]
