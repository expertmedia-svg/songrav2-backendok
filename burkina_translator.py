"""
burkina_translator.py
=====================
Moteur de traduction universel — Langues locales du Burkina Faso.

Priorites :
  1. Dictionnaire local CSV/JSON (confiance max)
  2. Recherche de synonymes dans le dictionnaire local
  3. Gemini AI avec regles grammaticales officielles + connaissances externes

Philosophie :
  - JAMAIS de traduction mot-a-mot (traduction du SENS, pas des mots)
  - Les mots absents du dict sont traites en silence — l'utilisateur ne voit rien
  - Chercher un synonyme avant de passer a l'IA
  - Respecter les normes de la Commission Nationale des Langues du Burkina Faso
"""

import os
import json
import re
import csv
from typing import Any, Dict, List, Optional

LANG_NAMES: Dict[str, str] = {
    "moore":     "Moore",
    "dioula":    "Dioula",
    "fulfulde":  "Fulfulde",
    "gourounsi": "Gourounsi",
    "bissa":     "Bissa",
}

VALID_LANGS = set(LANG_NAMES.keys())

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_TRANSLATE_DIR = os.path.join(_BACKEND_DIR, "TRANSLATE")

_DICT_FILES: Dict[str, str] = {
    "moore":     "dictionnaire_moore_1000.csv",
    "dioula":    "dictionnaire_dioula_1000.csv",
    "fulfulde":  "dictionnaire_fulfulde_1000.csv",
    "gourounsi": "dictionnaire_gourounsi_500.csv",
    "bissa":     "dictionnaire_bissa_500.csv",
}

_FR_SYNONYMS: Dict[str, List[str]] = {
    "maladie":      ["pathologie", "affection", "trouble"],
    "medicament":   ["remede", "traitement", "therapie"],
    "douleur":      ["mal", "souffrance", "gene"],
    "eau":          ["liquide"],
    "enfant":       ["bebe", "nourrisson", "jeune"],
    "medecin":      ["docteur", "praticien"],
    "champ":        ["terre", "parcelle", "culture"],
    "plante":       ["vegetal", "culture", "herbe"],
    "animal":       ["bete"],
    "urgent":       ["immediat", "critique", "grave"],
    "prevention":   ["protection", "securite"],
    "traitement":   ["soin", "remede", "therapie"],
    "diagnostic":   ["evaluation", "analyse", "bilan"],
    "symptome":     ["signe", "manifestation"],
    "infection":    ["contamination"],
    "consultation": ["visite", "examen"],
    "bonjour":      ["salut", "bonsoir", "salutation"],
    "merci":        ["remerciement", "gratitude"],
    "aide":         ["assistance", "soutien", "secours"],
}

_dictionaries: Dict[str, Dict[str, Any]] = {}


def _load_dict_from_path(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("{") or content.startswith("["):
            return json.loads(content)
        result: Dict[str, Any] = {}
        reader = csv.reader(content.splitlines())
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip().lower()
                val = row[1].strip()
                if key and val:
                    result[key] = {"translation": val}
        return result
    except Exception as e:
        print(f"[TRANSLATOR] Erreur chargement dict {path}: {e}")
        return {}


def _init_dictionaries() -> None:
    for lang, filename in _DICT_FILES.items():
        path = os.path.join(_TRANSLATE_DIR, filename)
        if not os.path.exists(path):
            path = os.path.join(_BACKEND_DIR, filename)
        loaded = _load_dict_from_path(path)
        _dictionaries[lang] = loaded
        count = len(loaded)
        status = f"{count} entrees" if count else "vide ou introuvable"
        print(f"[TRANSLATOR] Dict '{lang}' : {status} ({path})")


_init_dictionaries()


def _dict_lookup(word: str, lang: str) -> Optional[str]:
    d = _dictionaries.get(lang, {})
    key = word.lower().strip()
    entry = d.get(key)
    if entry:
        return entry.get("translation", "") if isinstance(entry, dict) else str(entry)
    for k, v in d.items():
        if key in k.split() or k in key.split():
            t = v.get("translation", "") if isinstance(v, dict) else str(v)
            if t:
                return t
    return None


def _synonym_lookup(word: str, lang: str) -> Optional[str]:
    import unicodedata
    def normalize(s):
        return unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode()
    key = normalize(word.strip())
    synonyms = _FR_SYNONYMS.get(key, [])
    for syn in synonyms:
        result = _dict_lookup(syn, lang)
        if result:
            return result
    for base_word, syns in _FR_SYNONYMS.items():
        if key in [normalize(s) for s in syns]:
            result = _dict_lookup(base_word, lang)
            if result:
                return result
    return None


def _build_dict_context(text: str, lang: str, max_entries: int = 25) -> Dict[str, Any]:
    d = _dictionaries.get(lang, {})
    words = re.sub(r"[.,!?;:()'\"\\/@]", " ", text.lower()).split()
    relevant: Dict[str, Any] = {}
    for w in words:
        if len(w) < 2:
            continue
        for k, v in d.items():
            if w in k or k in w:
                if len(relevant) < max_entries:
                    t = v.get("translation", "") if isinstance(v, dict) else str(v)
                    if t:
                        relevant[k] = t
    return relevant


_TRANSCRIPTION_RULES = (
    "NORMES DE TRANSCRIPTION (Commission Nationale des Langues du Burkina Faso) :\n"
    "- Alphabet officiel : utilisez les caracteres speciaux (epsilon, o ouvert) quand requis.\n"
    "- Nasalisation : inserez 'n' apres la voyelle nasalisee (ex: 'an', 'en', 'on').\n"
    "- Longueur vocalique : doublez la voyelle longue (ex: 'ee', 'oo', 'aa').\n"
    "- Tons : respectez les tons (haut/moyen/bas) dans le speech_text.\n"
    "- Emprunts : adaptez phonologiquement les termes modernes (ex: 'mobili' pour vehicule).\n"
    "- Ne jamais ecrire le mot francais brut pour un terme sans equivalent local."
)

_TRANSLATION_PRINCIPLES = (
    "PRINCIPES DE TRADUCTION (non negociables) :\n"
    "1. JAMAIS de traduction litterale mot-a-mot. Traduire le SENS, pas les mots.\n"
    "2. Si un mot exact n'existe pas, chercher un equivalent semantique ou une periphrase "
    "naturelle. L'utilisateur ne doit jamais voir de mots manquants ni de termes francais bruts.\n"
    "3. Utiliser les expressions idiomatiques locales du Burkina Faso.\n"
    "4. Adapter le registre au public rural burkinabe (langage oral, chaleureux, rassurant).\n"
    "5. Pour les termes techniques : utiliser l'equivalent culturel local si disponible, "
    "sinon adapter phonologiquement avec une periphrase explicative.\n"
    "6. Preferer les expressions culturellement equivalentes aux traductions approchantes."
)


def _call_gemini_translation(prompt: str, gemini_api_key: str) -> Optional[Dict[str, Any]]:
    import urllib.request
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(raw.strip())
    except Exception as e:
        print(f"[TRANSLATOR] Gemini echec: {e}")
        return None


def translate_text(
    text: str,
    target_lang: str,
    gemini_api_key: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"translation": "", "speech_text": "", "confidence": 1.0, "source": "empty"}
    if target_lang not in VALID_LANGS:
        return {"translation": text, "speech_text": text, "confidence": 0.0, "source": "invalid_lang"}

    lang_name = LANG_NAMES[target_lang]

    d = _dict_lookup(text, target_lang)
    if d:
        return {"translation": d, "speech_text": d, "confidence": 1.0, "source": "local_dictionary"}

    s = _synonym_lookup(text, target_lang)
    if s:
        return {"translation": s, "speech_text": s, "confidence": 0.85, "source": "local_synonym"}

    if gemini_api_key:
        dict_context = _build_dict_context(text, target_lang)
        dict_str = ""
        if dict_context:
            dict_str = (
                "\nExtrait du dictionnaire local (a respecter en priorite) :\n"
                + json.dumps(dict_context, ensure_ascii=False, indent=2) + "\n"
            )
        cat_ctx = f"Contexte : {category}. " if category else ""
        prompt = (
            f"Tu es un linguiste expert natif en {lang_name} (Burkina Faso).\n\n"
            f"{cat_ctx}Traduis le texte suivant du Francais vers le {lang_name} (code: {target_lang}).\n\n"
            f"{_TRANSCRIPTION_RULES}\n\n{_TRANSLATION_PRINCIPLES}\n{dict_str}\n"
            f"TEXTE A TRADUIRE :\n\"\"\"{text}\"\"\"\n\n"
            f"REPONSE JSON STRICTE :\n"
            "{{\n"
            f'  "translation": "<traduction naturelle en {lang_name}>",\n'
            '  "speech_text": "<version phonetique francisee pour TTS>",\n'
            '  "confidence": <0.0-1.0>\n'
            "}}"
        )
        ai = _call_gemini_translation(prompt, gemini_api_key)
        if ai and ai.get("translation"):
            return {
                "translation": ai.get("translation", text),
                "speech_text": ai.get("speech_text", ai.get("translation", text)),
                "confidence": float(ai.get("confidence", 0.75)),
                "source": "gemini_ai",
            }

    return {"translation": text, "speech_text": text, "confidence": 0.0, "source": "fallback_original"}


_SONGRA_TEXT_FIELDS = [
    "what_i_see", "disease_detected", "analysis", "detailed_analysis",
    "treatment", "treatment_local", "treatment_chemical", "prevention",
    "urgency_message", "recommendations", "consultation_type",
    "symptoms_observed", "diagnosis", "advice",
    "response", "answer", "message", "content", "summary",
    "resolution", "question", "description",
    "description_terrain", "decoupage_terrain", "gestion_eau",
    "title", "text", "body",
]


def translate_fields(
    fields: Dict[str, str],
    target_lang: str,
    gemini_api_key: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    if not fields or target_lang not in VALID_LANGS:
        return {}

    lang_name = LANG_NAMES[target_lang]
    results: Dict[str, Any] = {}
    needs_ai: Dict[str, str] = {}

    for field, text in fields.items():
        if not isinstance(text, str) or not text.strip():
            continue
        d = _dict_lookup(text, target_lang)
        if d:
            results[field] = {"translation": d, "speech_text": d, "confidence": 1.0, "source": "local_dictionary"}
            continue
        s = _synonym_lookup(text, target_lang)
        if s:
            results[field] = {"translation": s, "speech_text": s, "confidence": 0.85, "source": "local_synonym"}
            continue
        needs_ai[field] = text.strip()

    if needs_ai and gemini_api_key:
        dict_context = _build_dict_context(" ".join(needs_ai.values()), target_lang)
        dict_str = ""
        if dict_context:
            dict_str = (
                "\nExtrait du dictionnaire local (a respecter en priorite) :\n"
                + json.dumps(dict_context, ensure_ascii=False, indent=2) + "\n"
            )
        cat_ctx = f"Contexte : {category}. " if category else ""
        fields_json = json.dumps(needs_ai, ensure_ascii=False, indent=2)

        prompt = (
            f"Tu es un linguiste expert natif en {lang_name} (Burkina Faso).\n\n"
            f"{cat_ctx}Traduis chacun des champs suivants du Francais vers le {lang_name}.\n\n"
            f"{_TRANSCRIPTION_RULES}\n\n{_TRANSLATION_PRINCIPLES}\n{dict_str}\n"
            f"CHAMPS A TRADUIRE :\n{fields_json}\n\n"
            "REPONSE JSON STRICTE :\n"
            "{\n"
            '  "translations": {\n'
            f'    "<nom_champ>": {{"translation": "<texte en {lang_name}>", "speech_text": "<phonetique TTS>", "confidence": 0.0}}\n'
            "  }\n"
            "}"
        )
        ai_result = _call_gemini_translation(prompt, gemini_api_key)
        if ai_result and "translations" in ai_result:
            for field, res in ai_result["translations"].items():
                if isinstance(res, dict) and res.get("translation"):
                    results[field] = {
                        "translation": res.get("translation", needs_ai.get(field, "")),
                        "speech_text": res.get("speech_text", res.get("translation", "")),
                        "confidence": float(res.get("confidence", 0.75)),
                        "source": "gemini_ai_batch",
                    }
                    needs_ai.pop(field, None)

    for field, text in needs_ai.items():
        if field not in results:
            results[field] = {"translation": text, "speech_text": text, "confidence": 0.0, "source": "fallback_original"}

    return results


def translate_module_response(
    response: Dict[str, Any],
    target_lang: str,
    gemini_api_key: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Traduit automatiquement tous les champs textuels d'une reponse de module Songra.

    Applicable a TOUS les modules : scanner, assistant, teleconsultation,
    agriculture, veterinaire, SOS, communaute, entreprendre, etc.

    Ajoute un objet 'local_translation' a la reponse sans modifier l'original.
    Operation silencieuse — aucun crash possible, aucun mot manquant visible.
    """
    if not response or target_lang not in VALID_LANGS:
        return response

    to_translate: Dict[str, str] = {}
    for field in _SONGRA_TEXT_FIELDS:
        val = response.get(field)
        if isinstance(val, str) and val.strip():
            to_translate[field] = val.strip()
        elif isinstance(val, list):
            joined = " | ".join(str(v) for v in val if v)
            if joined:
                to_translate[field] = joined

    translated = translate_fields(to_translate, target_lang, gemini_api_key, category)

    result = dict(response)
    result["local_translation"] = {
        "target_lang": target_lang,
        "lang_name": LANG_NAMES[target_lang],
        "fields": translated,
    }
    return result
