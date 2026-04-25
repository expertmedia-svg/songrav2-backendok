"""
v2_services.py - Pipeline v2 unifié (porté de backend-node)
Fournisseur IA configurable via AI_PROVIDER (env var) : "openai" (défaut) ou "gemini"
Pour revenir à Gemini quand le billing est réglé : AI_PROVIDER=gemini dans .env

Modules :
- analyse unifiée texte + images (OpenAI GPT-4o ou Gemini)
- moteur de décision (image? vidéo? urgence?)
- générateur d'images (DALL-E 3 ou Gemini Imagen)
- générateur de vidéos (Veo via REST — Gemini uniquement)
- constructeur de réponse unique
- analyse entrepreneuriale (Entreprendre)
"""

import base64
import json
import os
import re
import time
import asyncio
from typing import Optional, List, Dict, Any

# ── OpenAI ──────────────────────────────────────────
try:
    from openai import OpenAI as _OpenAIClient
    _openai_available = True
except ImportError:
    _OpenAIClient = None
    _openai_available = False

# ── Gemini ──────────────────────────────────────────
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted

try:
    import google.genai as google_genai
    from google.genai import types as genai_types
except ImportError:
    google_genai = None
    genai_types = None

# ══════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════

# Fournisseur actif : "openai" ou "gemini"
# Changer dans .env (AI_PROVIDER=gemini) quand le billing Gemini est réglé
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower()

# ── OpenAI config ────────────────────────────────────
OPENAI_MODEL = "gpt-4o"
OPENAI_IMAGE_MODEL = "dall-e-3"
_openai_key = os.environ.get("OPENAI_API_KEY")
_openai_client: Optional[object] = _OpenAIClient(api_key=_openai_key) if (_openai_available and _openai_key) else None

def _get_openai_client():
    if not _openai_client:
        raise RuntimeError("OPENAI_API_KEY non définie ou openai non installé")
    return _openai_client

# ── Gemini config ────────────────────────────────────
_gemini_key = os.environ.get("GEMINI_API_KEY")
if _gemini_key:
    genai.configure(api_key=_gemini_key)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
VEO_MODEL = "veo-3.1-generate-preview"

# ── Commun ───────────────────────────────────────────
MAX_PHOTOS = 3
GEMINI_TIMEOUT = 60  # secondes (utilisé aussi pour OpenAI)
CACHE_TTL = 300  # 5 minutes

EMERGENCY_NUMBERS = {"pompiers": "18", "police": "17", "samu": "112"}

# Cache simple en mémoire
_analysis_cache: Dict[str, Dict] = {}

CATEGORY_EXPERTISE = {
    "agriculture": "agronome-conseil de terrain specialise des cultures vivrieres et maraicheres du Burkina Faso",
    "elevage": "agent veterinaire de terrain specialise en elevage rural au Burkina Faso",
    "urgence": "secouriste communautaire de terrain specialise en premiers gestes qui sauvent au Burkina Faso",
    "sos_accident": "secouriste communautaire de terrain specialise en premiers gestes qui sauvent au Burkina Faso",
    "cybersecurity": "conseiller local en securite numerique qui explique simplement les arnaques, piratages et bons reflexes mobiles",
}


# ══════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════

SYSTEM_PROMPT = """Tu es SONGRA, un assistant intelligent de terrain au Burkina Faso.
Tu combines les rôles de :
- Expert agricole africain (cultures sahéliennes : mil, sorgho, maïs, arachide, niébé, coton, sésame, oignon)
- Vétérinaire terrain (bovins, ovins, caprins, volailles, lapins)
- Secouriste d'urgence (premiers gestes, accidents, brûlures, morsures)

RÈGLES ABSOLUES :
1. Parle simplement, comme si tu parlais à un paysan qui n'a pas fait d'études
2. Donne des actions concrètes et réalisables avec les moyens locaux
3. Évite le jargon technique - utilise des mots du quotidien
4. Adapte-toi au climat sahélien, aux ressources limitées, au faible accès aux hôpitaux/vétérinaires
5. Priorise toujours les solutions locales avant les solutions chimiques
6. En cas d'urgence vitale, les gestes qui sauvent la vie passent EN PREMIER
7. Reste strictement dans la catégorie demandée : agriculture, élevage, urgence ou cybersécurité
8. Réponds comme un expert local crédible, habitué aux réalités des villages et quartiers du Burkina Faso
9. Quand tu proposes un visuel ou une vidéo, imagine toujours des personnes, vêtements, habitats et paysages cohérents avec le Burkina Faso rural ou périurbain
10. Les visuels et vidéos doivent être pédagogiques, calmes, non choquants et utiles pour apprendre un geste ou reconnaître un signe

CONTEXTE TERRAIN :
- Climat chaud et sec, saison des pluies courte
- Marchés locaux pour les produits de base
- Centres de santé parfois à des heures de route
- Eau propre pas toujours disponible
- Réseau mobile faible - réponses compactes nécessaires"""


def _category_expertise(category: str) -> str:
    return CATEGORY_EXPERTISE.get(category, "expert local de terrain du Burkina Faso")


def _category_context_hint(category: str) -> str:
    if category == "elevage":
        return (
            "CONTEXTE : problème lié à un animal d'élevage au Burkina Faso. "
            "Réponds comme un agent vétérinaire de proximité. "
            "Reste centré sur l'animal, les soins accessibles localement et le moment où il faut vite consulter."
        )
    if category in ("urgence", "sos_accident"):
        return (
            "CONTEXTE : situation d'urgence / accident / blessure humaine au Burkina Faso. "
            "Réponds comme un secouriste local. PRIORITE AUX GESTES QUI SAUVENT LA VIE. "
            "Aucun conseil dangereux, aucun geste complexe sans supervision."
        )
    if category == "cybersecurity":
        return (
            "CONTEXTE : sécurité numérique d'un utilisateur mobile au Burkina Faso. "
            "Réponds comme un conseiller local en cybersécurité. "
            "Reste centré sur les bons réflexes téléphone, WhatsApp, SMS, Mobile Money et mots de passe."
        )
    return (
        "CONTEXTE : problème agricole (culture, parcelle, sol) au Burkina Faso. "
        "Réponds comme un agronome de terrain. "
        "Reste centré sur les cultures, le sol, l'eau, les ravageurs et les traitements accessibles localement."
    )


def _visual_context_block(category: str, is_urgency: bool = False) -> str:
    base_context = (
        "Toujours situer la scene au Burkina Faso, dans un environnement rural ou periurbain credible. "
        "Montrer des personnes africaines, des habits simples du quotidien, des outils locaux, des concessions, champs, enclos ou centres de sante realistes. "
        "Le rendu doit etre pedagogique, calme, utile pour apprendre, sans element choquant ni sensationnaliste. "
        "Aucun texte incruste dans l'image ou la video."
    )
    if category == "elevage":
        return base_context + " Montrer un eleveur, un enclos propre, des animaux courants du Burkina Faso et des gestes de soin simples."
    if category in ("urgence", "sos_accident") or is_urgency:
        return base_context + " Montrer un cadre de premiers secours realiste, propre et non graphique, avec des gestes lents et rassurants."
    if category == "cybersecurity":
        return base_context + " Si un visuel est demande, montrer un telephone, un SMS, une interface Mobile Money ou WhatsApp de facon simple et non anxiogene."
    return base_context + " Montrer des cultures, outils et gestes agricoles realistes pour le Burkina Faso."

ANALYSIS_PROMPT = SYSTEM_PROMPT + """

TÂCHE : Analyser le problème décrit (texte et/ou image) et retourner un diagnostic structuré.

Si une image est fournie, commence par décrire CE QUE TU VOIS avant de diagnostiquer.
Si du texte est fourni, utilise-le comme contexte supplémentaire.

RETOURNE UNIQUEMENT un objet JSON valide avec cette structure EXACTE :
{
  "type_probleme": "agriculture | elevage | urgence",
  "description_visuelle": "ce que tu observes sur l'image (vide si pas d'image)",
  "diagnostic": "explication claire et simple du problème identifié",
  "gravite": "faible | moyenne | critique",
  "confiance": 0.0 à 1.0,
  "causes_probables": ["cause 1", "cause 2"],
  "actions_immediates": ["action urgente 1", "action urgente 2"],
  "actions_detaillees": ["étape détaillée 1 avec dosages si applicable", "étape 2"],
  "actions_preventives": ["prévention 1", "prévention 2"],
  "besoin_image": true ou false (true si une image explicative aiderait),
  "besoin_video": true ou false (true si une vidéo pédagogique aiderait),
  "consulter_expert": true ou false,
  "message_expert": "quand et pourquoi consulter un expert (vide si pas nécessaire)"
}

IMPORTANT :
- Retourne UNIQUEMENT le JSON, pas de texte avant ni après
- Pas de commentaires, pas de markdown, juste le JSON pur
- En cas d'urgence vitale (gravite = "critique"), les actions_immediates doivent être des gestes de survie numérotés
- Le diagnostic et les actions doivent rester strictement dans la catégorie demandée et refléter les réalités du Burkina Faso"""


ENTREPRENDRE_PROMPT = SYSTEM_PROMPT + """

TÂCHE : Tu es maintenant en mode CONSEIL ENTREPRENEURIAL RURAL.
L'utilisateur te montre une photo de son terrain, parcelle, champ, ou espace d'élevage.
Il veut savoir COMMENT exploiter ce terrain pour gagner de l'argent et nourrir sa famille.

Si une image est fournie, commence par décrire CE QUE TU VOIS (type de sol, végétation, relief, eau visible, surface estimée, état du terrain).

RETOURNE UNIQUEMENT un objet JSON valide avec cette structure EXACTE :
{
  "description_terrain": "description détaillée de ce que tu observes sur le terrain",
  "surface_estimee": "estimation de la surface visible (ex: 0.5 hectare)",
  "type_sol": "type de sol estimé (argileux, sableux, latéritique, alluvial, etc.)",
  "potentiel": "faible | moyen | bon | excellent",
  "propositions": [
    {
      "titre": "nom du projet (ex: Culture de niébé + maraîchage)",
      "description": "explication simple de ce qu'il faut faire",
      "investissement": "estimation du coût de départ en FCFA",
      "revenu_estime": "estimation du revenu par saison/mois en FCFA",
      "duree_retour": "temps avant les premiers revenus (ex: 3 mois)",
      "difficulte": "facile | moyen | avancé"
    }
  ],
  "decoupage_terrain": "comment diviser/organiser le terrain",
  "calendrier_cultural": [
    {
      "mois": "juin-juillet",
      "activite": "préparer le sol, labourer",
      "details": "ce qu'il faut faire exactement"
    }
  ],
  "gestion_eau": {
    "sources": ["puits", "collecte pluie"],
    "techniques": ["zaï", "demi-lune", "cordons pierreux"],
    "conseils": "comment préserver l'eau sur ce terrain"
  },
  "engrais_et_semences": {
    "quand_semer": "période optimale de semis",
    "quand_engrais": "timing des apports",
    "types_engrais": ["fumier", "compost", "NPK si budget"],
    "semences_recommandees": ["variétés adaptées au sahel"]
  },
  "astuces_locales": ["astuce 1", "astuce 2"],
  "risques": ["risque 1", "risque 2"],
  "besoin_image": true,
  "besoin_video": true
}

IMPORTANT :
- Propose au moins 2-3 projets différents adaptés au terrain visible
- Privilégie les solutions locales, peu coûteuses, accessibles aux paysans
- Inclus toujours un projet à faible investissement (< 25000 FCFA)
- Donne des chiffres réalistes pour le Burkina Faso
- Le calendrier doit suivre les saisons du Sahel (sèche oct-mai, pluies juin-sept)
- Si tu proposes un visuel ou une vidéo, ils doivent montrer un terrain et des personnes cohérents avec le Burkina Faso
- Retourne UNIQUEMENT le JSON, pas de texte avant ni après"""


# ══════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════

def _get_model(model_name: str = GEMINI_MODEL):
    """Obtenir un modèle Gemini configuré"""
    return genai.GenerativeModel(model_name)


async def _generate_content_with_timeout(
    model,
    prompt_or_parts,
    *,
    timeout: int = GEMINI_TIMEOUT,
):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt_or_parts),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Gemini a depasse le delai autorise de {timeout} secondes"
        ) from exc


def _get_media_client(api_key: Optional[str] = None):
    """Client google.genai dédié aux générations image/vidéo."""
    if google_genai is None or genai_types is None:
        raise RuntimeError("google-genai indisponible dans cet environnement")

    key = api_key or os.environ.get("GEMINI_API_KEY") or _gemini_key
    if not key:
        raise ValueError("GEMINI_API_KEY non definie")
    return google_genai.Client(api_key=key)


def _image_aspect_ratio(style: str) -> str:
    if style == "schema":
        return "1:1"
    if style == "photo_realiste":
        return "16:9"
    return "4:3"


def _sanitize_visual_prompt(prompt: str, is_urgency: bool) -> str:
    cleaned = " ".join((prompt or "").split())
    if not is_urgency:
        return cleaned
    return (
        "Illustration ou animation pédagogique non graphique de premiers secours. "
        "Ne montrer ni sang, ni plaie ouverte, ni blessure choquante. "
        "Montrer seulement les mains, le tissu propre, le bandage, la position du corps et le geste sûr. "
        f"Sujet: {cleaned}"
    )


def _normalize_video_duration(duration_sec: int) -> int:
    if duration_sec <= 4:
        return 4
    if duration_sec <= 6:
        return 6
    return 8


def _parse_gemini_json(response_text: str) -> dict:
    """Parse robuste du JSON retourné par Gemini"""
    cleaned = re.sub(r"```json\s*", "", response_text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                sanitized = json_match.group(0)
                sanitized = re.sub(r"[\r\n]+", " ", sanitized)
                sanitized = re.sub(r",\s*}", "}", sanitized)
                sanitized = re.sub(r",\s*]", "]", sanitized)
                return json.loads(sanitized)
    raise ValueError("Impossible de parser la réponse Gemini en JSON")


def _validate_analysis(raw: dict) -> dict:
    """Valide et normalise le JSON d'analyse"""
    valid_types = ["agriculture", "elevage", "urgence"]
    valid_gravite = ["faible", "moyenne", "critique"]

    return {
        "type_probleme": raw.get("type_probleme") if raw.get("type_probleme") in valid_types else "agriculture",
        "description_visuelle": str(raw.get("description_visuelle") or ""),
        "diagnostic": str(raw.get("diagnostic") or "Diagnostic non disponible"),
        "gravite": raw.get("gravite") if raw.get("gravite") in valid_gravite else "moyenne",
        "confiance": max(0.0, min(1.0, float(raw.get("confiance") or 0.5))),
        "causes_probables": [str(c) for c in raw.get("causes_probables", [])],
        "actions_immediates": [str(a) for a in raw.get("actions_immediates", [])],
        "actions_detaillees": [str(a) for a in raw.get("actions_detaillees", [])],
        "actions_preventives": [str(a) for a in raw.get("actions_preventives", [])],
        "besoin_image": bool(raw.get("besoin_image")),
        "besoin_video": bool(raw.get("besoin_video")),
        "consulter_expert": bool(raw.get("consulter_expert")),
        "message_expert": str(raw.get("message_expert") or ""),
    }


def _validate_entrepreneurship(raw: dict) -> dict:
    """Valide et normalise le JSON entrepreneuriat"""
    valid_potentiel = ["faible", "moyen", "bon", "excellent"]
    valid_difficulte = ["facile", "moyen", "avancé"]

    return {
        "description_terrain": str(raw.get("description_terrain") or ""),
        "surface_estimee": str(raw.get("surface_estimee") or "Non estimée"),
        "type_sol": str(raw.get("type_sol") or "Non déterminé"),
        "potentiel": raw.get("potentiel") if raw.get("potentiel") in valid_potentiel else "moyen",
        "propositions": [
            {
                "titre": str(p.get("titre") or ""),
                "description": str(p.get("description") or ""),
                "investissement": str(p.get("investissement") or ""),
                "revenu_estime": str(p.get("revenu_estime") or ""),
                "duree_retour": str(p.get("duree_retour") or ""),
                "difficulte": p.get("difficulte") if p.get("difficulte") in valid_difficulte else "moyen",
            }
            for p in (raw.get("propositions") or [])
        ],
        "decoupage_terrain": str(raw.get("decoupage_terrain") or ""),
        "calendrier_cultural": [
            {
                "mois": str(c.get("mois") or ""),
                "activite": str(c.get("activite") or ""),
                "details": str(c.get("details") or ""),
            }
            for c in (raw.get("calendrier_cultural") or [])
        ],
        "gestion_eau": {
            "sources": [str(s) for s in (raw.get("gestion_eau") or {}).get("sources", [])],
            "techniques": [str(t) for t in (raw.get("gestion_eau") or {}).get("techniques", [])],
            "conseils": str((raw.get("gestion_eau") or {}).get("conseils") or ""),
        },
        "engrais_et_semences": {
            "quand_semer": str((raw.get("engrais_et_semences") or {}).get("quand_semer") or ""),
            "quand_engrais": str((raw.get("engrais_et_semences") or {}).get("quand_engrais") or ""),
            "types_engrais": [str(t) for t in (raw.get("engrais_et_semences") or {}).get("types_engrais", [])],
            "semences_recommandees": [str(s) for s in (raw.get("engrais_et_semences") or {}).get("semences_recommandees", [])],
        },
        "astuces_locales": [str(a) for a in (raw.get("astuces_locales") or [])],
        "risques": [str(r) for r in (raw.get("risques") or [])],
        "besoin_image": raw.get("besoin_image") is not False,
        "besoin_video": raw.get("besoin_video") is not False,
    }


def _get_cache_key(text: str, category: str, has_image: bool) -> str:
    normalized = (text or "").lower().strip()[:200]
    return f"{category}:{'img' if has_image else 'txt'}:{normalized}"


# ══════════════════════════════════════════════════════
# OPENAI PROVIDER — fonctions internes
# ══════════════════════════════════════════════════════

async def _openai_chat(system: str, user_text: str, images_b64: Optional[List[str]] = None, max_tokens: int = 2000) -> str:
    """Appel OpenAI chat completions avec support vision"""
    client = _get_openai_client()
    images_b64 = images_b64 or []

    user_content: list = [{"type": "text", "text": user_text}]
    for img_b64 in images_b64[:MAX_PHOTOS]:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    response = await asyncio.wait_for(
        asyncio.to_thread(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
        ),
        timeout=GEMINI_TIMEOUT,
    )
    return response.choices[0].message.content


async def _openai_analyze(text: str, images_b64: List[str], category: str) -> dict:
    """Analyse unifiée via OpenAI GPT-4o (remplace gemini_analyze quand AI_PROVIDER=openai)"""
    has_image = len(images_b64) > 0
    has_text = bool(text.strip())

    context_hint = f"\n{_category_context_hint(category)}"
    expertise_hint = f"\nPOSTURE : parle comme un {_category_expertise(category)}."
    user_message = (
        f"\nDescription de l'utilisateur : {text}" if has_text
        else "\n(Pas de description textuelle - analyse basée sur l'image uniquement)"
    )
    full_prompt = ANALYSIS_PROMPT + context_hint + expertise_hint + user_message

    try:
        response_text = await _openai_chat(SYSTEM_PROMPT, full_prompt, images_b64)
        raw_json = _parse_gemini_json(response_text)
        analysis = _validate_analysis(raw_json)
    except Exception as error:
        print(f"[openai_analyze] Fallback local activé: {error}")
        analysis = _build_analysis_fallback(text=text, category=category, has_image=has_image, error=error)

    if category in ("urgence", "sos_accident"):
        analysis["type_probleme"] = "urgence"
    return analysis


async def _openai_analyze_entrepreneurship(text: str, images_b64: List[str], category: str) -> dict:
    """Analyse entrepreneuriale via OpenAI GPT-4o"""
    has_image = len(images_b64) > 0
    has_text = bool(text.strip())

    if category == "elevage":
        context_hint = "\nCONTEXTE : terrain destiné à l'élevage au Burkina Faso. Propose des projets d'élevage ET de culture mixte."
    else:
        context_hint = "\nCONTEXTE : terrain agricole au Burkina Faso. Propose des projets de culture ET éventuellement d'élevage complémentaire."

    user_message = (
        f"\nDescription de l'utilisateur : {text}" if has_text
        else "\n(Pas de description textuelle - analyse basée sur l'image uniquement)"
    )
    full_prompt = ENTREPRENDRE_PROMPT + context_hint + user_message

    try:
        response_text = await _openai_chat(SYSTEM_PROMPT, full_prompt, images_b64)
        raw_json = _parse_gemini_json(response_text)
        return _validate_entrepreneurship(raw_json)
    except Exception as error:
        print(f"[openai_analyze_entrepreneurship] Fallback local activé: {error}")
        return _build_entrepreneurship_fallback(category=category, has_image=has_image, error=error)


async def _openai_generate_image(prompt: str, style: str = "illustration", category: str = "agriculture") -> dict:
    """Génère une image via DALL-E 3 (remplace Gemini Imagen quand AI_PROVIDER=openai)"""
    try:
        client = _get_openai_client()
        size_map = {"photo_realiste": "1792x1024", "schema": "1024x1024", "illustration": "1024x1024"}
        size = size_map.get(style, "1024x1024")

        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.images.generate,
                model=OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size=size,
                response_format="b64_json",
                n=1,
            ),
            timeout=60,
        )
        image_b64 = response.data[0].b64_json
        return {"success": True, "image_base64": image_b64, "mime_type": "image/png"}
    except Exception as e:
        print(f"[openai_generate_image] Erreur: {e}")
        return {"success": False, "error": str(e), "fallback_description": prompt}


async def _openai_llm_answer(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Réponse LLM texte via OpenAI GPT-4o"""
    try:
        return await _openai_chat(system_prompt, user_prompt)
    except Exception as e:
        print(f"[openai_llm_answer] Erreur: {e}")
        return None


def _should_use_resilient_fallback(error: Exception) -> bool:
    if isinstance(error, (ResourceExhausted, GoogleAPICallError, TimeoutError)):
        return True

    message = str(error).lower()
    fallback_markers = [
        "spending cap",
        "resourceexhausted",
        "quota",
        "rate limit",
        "429",
        "service unavailable",
        "deadline exceeded",
        "timed out",
        "impossible de parser la réponse gemini",
    ]
    return any(marker in message for marker in fallback_markers)


def _build_analysis_fallback(text: str, category: str, has_image: bool, error: Exception) -> dict:
    normalized_text = (text or "").lower()
    type_probleme = "urgence" if category in ("urgence", "sos_accident") else category
    description_visuelle = "Photo reçue mais analyse visuelle avancée temporairement indisponible." if has_image else ""

    if type_probleme == "urgence":
        return {
            "type_probleme": "urgence",
            "description_visuelle": description_visuelle,
            "diagnostic": "Mode secours activé: la situation doit être évaluée prudemment. En cas de douleur forte, saignement, perte de connaissance, difficulté à respirer ou aggravation rapide, cherchez immédiatement une aide médicale.",
            "gravite": "critique",
            "confiance": 0.35,
            "causes_probables": [
                "Analyse IA distante temporairement indisponible",
                "Le problème exact ne peut pas être confirmé sans examen plus précis",
            ],
            "actions_immediates": [
                "Mettre la personne en sécurité et l'éloigner du danger immédiat.",
                "Observer respiration, conscience et saignement.",
                "Appeler ou rejoindre rapidement un centre de santé si l'état paraît grave.",
            ],
            "actions_detaillees": [
                "Utiliser seulement de l'eau propre et un tissu propre pour protéger la zone touchée.",
                "Éviter les gestes risqués ou les produits agressifs sans avis soignant.",
            ],
            "actions_preventives": [
                "Refaire une demande avec plus de détails sur l'accident ou une photo plus nette si possible.",
            ],
            "besoin_image": False,
            "besoin_video": False,
            "consulter_expert": True,
            "message_expert": "Réponse de secours utilisée car le service IA avancé est momentanément indisponible.",
            "from_fallback": True,
            "fallback_reason": str(error),
        }

    if type_probleme == "elevage":
        return {
            "type_probleme": "elevage",
            "description_visuelle": description_visuelle,
            "diagnostic": "Mode secours activé: le problème semble concerner la santé ou l'état de l'animal, mais le diagnostic précis ne peut pas être confirmé pour le moment.",
            "gravite": "moyenne",
            "confiance": 0.4,
            "causes_probables": [
                "Infection ou blessure locale",
                "Stress, chaleur ou hygiène insuffisante",
                "Analyse IA distante temporairement indisponible",
            ],
            "actions_immediates": [
                "Isoler l'animal si son état semble contagieux ou s'il est très affaibli.",
                "Vérifier s'il mange, boit, boite, saigne ou respire mal.",
            ],
            "actions_detaillees": [
                "Nettoyer doucement la zone anormale si elle est visible, avec un produit adapté si disponible.",
                "Garder l'abri propre, sec et calme.",
                "Refaire une photo nette de la zone touchée et décrire les signes observés.",
            ],
            "actions_preventives": [
                "Surveiller les autres animaux du lot.",
                "Désinfecter le matériel de soin et les abreuvoirs si nécessaire.",
            ],
            "besoin_image": False,
            "besoin_video": False,
            "consulter_expert": True,
            "message_expert": "Réponse de secours utilisée car le service IA avancé est momentanément indisponible.",
            "from_fallback": True,
            "fallback_reason": str(error),
        }

    causes = [
        "Carence nutritive ou épuisement du sol",
        "Stress hydrique ou drainage insuffisant",
        "Début de maladie foliaire ou attaque de ravageurs",
    ]
    diagnostic = "Mode secours activé: la culture montre un stress végétal qui demande vérification locale."
    immediate_actions = [
        "Observer si le problème touche seulement quelques plants ou toute la parcelle.",
        "Vérifier l'humidité du sol au pied des plants.",
    ]
    detailed_actions = [
        "Retirer les feuilles très abîmées si elles sont sèches ou fortement tachées.",
        "Éviter d'arroser le feuillage tard le soir et privilégier l'arrosage au pied.",
        "Refaire une photo nette d'une feuille entière et du pied du plant.",
    ]
    preventive_actions = [
        "Surveiller l'évolution sur 24 à 48 heures.",
        "Prévoir un apport organique ou un complément adapté si le sol paraît pauvre.",
    ]

    if any(marker in normalized_text for marker in ["jaune", "jaun", "feuille"]):
        diagnostic = "Le jaunissement des feuilles évoque surtout un stress de culture, souvent lié à une carence nutritive, un manque ou excès d'eau, ou un début de maladie foliaire."
        causes = [
            "Carence en azote ou sol appauvri",
            "Excès d'eau ou manque d'eau",
            "Début de maladie sur les feuilles",
        ]
        detailed_actions.insert(0, "Comparer les feuilles du bas et du haut pour voir si le jaunissement commence par la base.")

    return {
        "type_probleme": "agriculture",
        "description_visuelle": description_visuelle,
        "diagnostic": diagnostic,
        "gravite": "moyenne",
        "confiance": 0.42,
        "causes_probables": causes,
        "actions_immediates": immediate_actions,
        "actions_detaillees": detailed_actions,
        "actions_preventives": preventive_actions,
        "besoin_image": False,
        "besoin_video": False,
        "consulter_expert": False,
        "message_expert": "Réponse de secours utilisée car le service IA avancé est momentanément indisponible.",
        "from_fallback": True,
        "fallback_reason": str(error),
    }


def _build_entrepreneurship_fallback(category: str, has_image: bool, error: Exception) -> dict:
    if category == "elevage":
        propositions = [
            {
                "titre": "Petit élevage de volailles amélioré",
                "description": "Commencer avec un petit lot, un abri propre, eau et alimentation régulières.",
                "investissement": "20000 à 50000 FCFA",
                "revenu_estime": "15000 à 40000 FCFA par cycle",
                "duree_retour": "2 à 3 mois",
                "difficulte": "facile",
            },
            {
                "titre": "Embouche de petits ruminants",
                "description": "Valoriser un petit espace avec quelques animaux et une alimentation suivie.",
                "investissement": "50000 à 150000 FCFA",
                "revenu_estime": "30000 à 80000 FCFA selon le cycle",
                "duree_retour": "3 à 6 mois",
                "difficulte": "moyen",
            },
        ]
    else:
        propositions = [
            {
                "titre": "Maraîchage à petite échelle",
                "description": "Exploiter une petite surface avec des cultures à rotation rapide et arrosage maîtrisé.",
                "investissement": "15000 à 40000 FCFA",
                "revenu_estime": "20000 à 60000 FCFA par cycle",
                "duree_retour": "1 à 3 mois",
                "difficulte": "facile",
            },
            {
                "titre": "Culture vivrière avec compost local",
                "description": "Associer culture principale et apport organique pour réduire les coûts.",
                "investissement": "10000 à 30000 FCFA",
                "revenu_estime": "25000 à 70000 FCFA par saison",
                "duree_retour": "3 à 5 mois",
                "difficulte": "moyen",
            },
        ]

    return {
        "description_terrain": "Mode secours activé: le terrain ne peut pas être analysé finement pour le moment, mais des pistes locales à faible coût restent proposées.",
        "surface_estimee": "Non estimée",
        "type_sol": "À confirmer sur place",
        "potentiel": "moyen",
        "propositions": propositions,
        "decoupage_terrain": "Prévoir une petite zone de test, une zone principale de production et un espace de stockage/eau si possible.",
        "calendrier_cultural": [
            {"mois": "début de saison", "activite": "préparer le terrain", "details": "Nettoyer, délimiter et sécuriser l'eau ou l'abri."},
            {"mois": "mise en place", "activite": "lancer un petit projet pilote", "details": "Commencer petit pour mesurer coûts, mortalité ou rendement."},
        ],
        "gestion_eau": {
            "sources": ["pluie", "réserve locale", "puits si disponible"],
            "techniques": ["paillage", "arrosage ciblé", "récupération d'eau"],
            "conseils": "Sécuriser d'abord une solution d'eau minimale avant d'agrandir le projet.",
        },
        "engrais_et_semences": {
            "quand_semer": "Selon la pluie et la disponibilité en eau",
            "quand_engrais": "Après reprise correcte des plants ou du projet",
            "types_engrais": ["compost", "fumier bien décomposé"],
            "semences_recommandees": ["variétés locales adaptées"],
        },
        "astuces_locales": [
            "Commencer petit pour valider la rentabilité avant d'investir davantage.",
            "Choisir un projet facile à suivre avec les moyens réellement disponibles.",
        ],
        "risques": [
            "Manque d'eau ou alimentation insuffisante",
            "Sous-estimation des coûts de départ",
            "Analyse IA avancée temporairement indisponible",
        ],
        "besoin_image": False,
        "besoin_video": False,
        "from_fallback": True,
        "fallback_reason": str(error),
    }


# ══════════════════════════════════════════════════════
# SERVICE D'ANALYSE GEMINI UNIFIÉ
# ══════════════════════════════════════════════════════

async def gemini_analyze(
    text: str = "",
    images_b64: Optional[List[str]] = None,
    category: str = "agriculture",
) -> dict:
    """Analyse unifiée texte + images → JSON structuré.
    Dispatche vers OpenAI ou Gemini selon AI_PROVIDER."""
    images_b64 = images_b64 or []
    has_image = len(images_b64) > 0
    has_text = bool(text.strip())

    if not has_text and not has_image:
        raise ValueError("Veuillez fournir au moins du texte ou une image")

    # ── Routing provider ──────────────────────────────
    if AI_PROVIDER == "openai":
        # Cache (texte seul) aussi pour OpenAI
        if not has_image and has_text:
            cache_key = _get_cache_key(text, category, False)
            cached = _analysis_cache.get(cache_key)
            if cached and (time.time() - cached["ts"]) < CACHE_TTL:
                return {**cached["data"], "from_cache": True}
        analysis = await _openai_analyze(text, images_b64, category)
        if not has_image and has_text:
            cache_key = _get_cache_key(text, category, False)
            _analysis_cache[cache_key] = {"data": analysis, "ts": time.time()}
            if len(_analysis_cache) > 500:
                del _analysis_cache[next(iter(_analysis_cache))]
        return analysis

    # ── Gemini (réactiver via AI_PROVIDER=gemini) ─────
    # Cache (texte seul)
    if not has_image and has_text:
        cache_key = _get_cache_key(text, category, False)
        cached = _analysis_cache.get(cache_key)
        if cached and (time.time() - cached["ts"]) < CACHE_TTL:
            return {**cached["data"], "from_cache": True}

    model = _get_model()

    context_hint = f"\n{_category_context_hint(category)}"
    expertise_hint = f"\nPOSTURE : parle comme un {_category_expertise(category)}."

    user_message = f"\nDescription de l'utilisateur : {text}" if has_text else "\n(Pas de description textuelle - analyse basée sur l'image uniquement)"
    full_prompt = ANALYSIS_PROMPT + context_hint + expertise_hint + user_message

    content_parts = [full_prompt]
    for img_b64 in images_b64[:MAX_PHOTOS]:
        content_parts.append({
            "mime_type": "image/jpeg",
            "data": img_b64,
        })

    try:
        result = await _generate_content_with_timeout(model, content_parts)
        response_text = result.text
        raw_json = _parse_gemini_json(response_text)
        analysis = _validate_analysis(raw_json)
    except Exception as error:
        if not _should_use_resilient_fallback(error):
            raise
        print(f"[gemini_analyze] Fallback local active: {error}")
        analysis = _build_analysis_fallback(text=text, category=category, has_image=has_image, error=error)

    if category in ("urgence", "sos_accident"):
        analysis["type_probleme"] = "urgence"

    # Mettre en cache (texte seul)
    if not has_image and has_text:
        cache_key = _get_cache_key(text, category, False)
        _analysis_cache[cache_key] = {"data": analysis, "ts": time.time()}
        if len(_analysis_cache) > 500:
            oldest = next(iter(_analysis_cache))
            del _analysis_cache[oldest]

    return analysis


async def gemini_analyze_entrepreneurship(
    text: str = "",
    images_b64: Optional[List[str]] = None,
    category: str = "agriculture",
) -> dict:
    """Analyse entrepreneuriale terrain.
    Dispatche vers OpenAI ou Gemini selon AI_PROVIDER."""
    images_b64 = images_b64 or []
    has_image = len(images_b64) > 0
    has_text = bool(text.strip())

    if not has_text and not has_image:
        raise ValueError("Envoyez une photo de votre terrain ou décrivez-le")

    # ── Routing provider ──────────────────────────────
    if AI_PROVIDER == "openai":
        return await _openai_analyze_entrepreneurship(text, images_b64, category)

    # ── Gemini (réactiver via AI_PROVIDER=gemini) ─────
    model = _get_model()

    if category == "elevage":
        context_hint = "\nCONTEXTE : terrain destiné à l'élevage au Burkina Faso. Propose des projets d'élevage ET de culture mixte. Réponds comme un conseiller local qui connaît les marchés, l'eau et les contraintes réelles du terrain."
    else:
        context_hint = "\nCONTEXTE : terrain agricole au Burkina Faso. Propose des projets de culture ET éventuellement d'élevage complémentaire. Réponds comme un conseiller local qui connaît les saisons, les sols et les débouchés réels."

    user_message = f"\nDescription de l'utilisateur : {text}" if has_text else "\n(Pas de description textuelle - analyse basée sur l'image uniquement)"
    full_prompt = ENTREPRENDRE_PROMPT + context_hint + user_message

    content_parts = [full_prompt]
    for img_b64 in images_b64[:MAX_PHOTOS]:
        content_parts.append({
            "mime_type": "image/jpeg",
            "data": img_b64,
        })

    try:
        result = await _generate_content_with_timeout(model, content_parts)
        response_text = result.text
        raw_json = _parse_gemini_json(response_text)
        return _validate_entrepreneurship(raw_json)
    except Exception as error:
        if not _should_use_resilient_fallback(error):
            raise
        print(f"[gemini_analyze_entrepreneurship] Fallback local active: {error}")
        return _build_entrepreneurship_fallback(category=category, has_image=has_image, error=error)


# ══════════════════════════════════════════════════════
# MOTEUR DE DÉCISION
# ══════════════════════════════════════════════════════

def decide(analysis: dict) -> dict:
    """Décide quoi faire basé sur le diagnostic structuré"""
    gravite = analysis.get("gravite", "moyenne")
    confiance = analysis.get("confiance", 0.5)
    type_probleme = analysis.get("type_probleme", "agriculture")
    consulter_expert = analysis.get("consulter_expert", False)

    decision = {
        "mode_urgence": False,
        "generer_image": False,
        "prompt_image": None,
        "generer_video": False,
        "prompt_video": None,
        "format_reponse": "standard",
        "transfert_expert": False,
        "priorite": 3,
    }

    # MODE URGENCE
    if gravite == "critique" or type_probleme == "urgence":
        decision["mode_urgence"] = True
        decision["format_reponse"] = "urgence"
        decision["priorite"] = 1
        decision["generer_image"] = True
        decision["prompt_image"] = (
            f"Illustration simple et claire montrant le geste de premier secours pour : {analysis['diagnostic']}. "
            "Style schématique, couleurs vives, compréhensible sans savoir lire. "
            "Contexte : Burkina Faso, premiers secours locaux, posture calme et pédagogique."
        )
        decision["generer_video"] = True
        actions_text = ". ".join(analysis.get("actions_immediates", []))
        decision["prompt_video"] = (
            f"Vidéo courte de premiers secours (5-8 secondes) montrant les gestes d'urgence pour : {analysis['diagnostic']}. "
            f"Actions : {actions_text}. Style simple, contexte : Burkina Faso, village ou quartier local, démonstration pédagogique non graphique."
        )
        return decision

    # GRAVITÉ MOYENNE
    if gravite == "moyenne":
        decision["priorite"] = 2
        decision["format_reponse"] = "detaille"
        if confiance < 0.5 or consulter_expert:
            decision["transfert_expert"] = True

    # GRAVITÉ FAIBLE
    if consulter_expert:
        decision["transfert_expert"] = True

    # En mode standard, on garde l'image auto mais la vidéo reste à la demande
    # via /api/v2/generate-video-illustration pour éviter de bloquer l'analyse.
    decision["generer_image"] = True
    decision["generer_video"] = False

    if type_probleme == "agriculture":
        decision["prompt_image"] = (
            f"Illustration pédagogique agricole montrant : {analysis['diagnostic']}. "
            "Montrer les symptômes sur la plante et le traitement recommandé. "
            "Style dessin simple, adapté pour des agriculteurs du Burkina Faso."
        )
        actions_text = ". ".join(analysis.get("actions_detaillees", [])[:3])
        decision["prompt_video"] = (
            f"Vidéo pédagogique courte (5-10 secondes) montrant les gestes techniques pour traiter : {analysis['diagnostic']}. "
            f"Étapes : {actions_text}. Contexte : champ sahélien au Burkina Faso, démonstration locale claire."
        )
    elif type_probleme == "elevage":
        decision["prompt_image"] = (
            f"Illustration vétérinaire simple montrant : {analysis['diagnostic']}. "
            "Montrer les signes à observer sur l'animal et les soins de base. "
            "Style schématique, adapté aux éleveurs ruraux du Burkina Faso."
        )
        actions_text = ". ".join(analysis.get("actions_detaillees", [])[:3])
        decision["prompt_video"] = (
            f"Vidéo pédagogique courte (5-10 secondes) montrant les soins de base pour : {analysis['diagnostic']}. "
            f"Étapes : {actions_text}. Contexte : élevage rural au Burkina Faso, gestes simples et locaux."
        )
    else:
        decision["prompt_image"] = (
            f"Illustration explicative simple pour : {analysis['diagnostic']}. "
            "Style clair, compréhensible par tous, ancré dans le Burkina Faso."
        )
        actions_text = ". ".join(analysis.get("actions_detaillees", [])[:3])
        decision["prompt_video"] = (
            f"Vidéo pédagogique courte (5-10 secondes) pour : {analysis['diagnostic']}. "
            f"Étapes : {actions_text}. Contexte local, démonstration simple et utile."
        )

    if not decision["generer_video"]:
        decision["prompt_video"] = None

    return decision


# ══════════════════════════════════════════════════════
# GÉNÉRATEUR D'IMAGES (Gemini)
# ══════════════════════════════════════════════════════

async def generate_image(prompt: str, style: str = "illustration", category: str = "agriculture") -> dict:
    """Génère une image explicative (DALL-E 3 ou Gemini Imagen selon AI_PROVIDER)"""
    if not prompt or not prompt.strip():
        return {"success": False, "error": "Prompt image requis"}

    # ── Routing provider ──────────────────────────────
    if AI_PROVIDER == "openai":
        is_urgency = style == "schema"
        visual_prompt = _sanitize_visual_prompt(prompt, is_urgency=is_urgency)
        style_instructions = {
            "schema": "Style schématique simple, fond blanc, traits noirs épais, couleurs vives. Compréhensible sans savoir lire.",
            "illustration": "Style illustration pédagogique, couleurs chaudes, personnages africains, paysage sahélien.",
            "photo_realiste": "Style photo-réaliste, contexte rural africain, éclairage naturel.",
        }
        enriched = (
            f"{visual_prompt} {style_instructions.get(style, '')} "
            f"{_visual_context_block(category, is_urgency=is_urgency)} "
            "Pas de texte dans l'image."
        )
        return await _openai_generate_image(enriched, style, category)

    style_instructions = {
        "schema": "Style schématique simple, fond blanc, traits noirs épais, couleurs vives de base (rouge, vert, jaune). Compréhensible sans savoir lire.",
        "illustration": "Style illustration pédagogique, couleurs chaudes, personnages africains, paysage sahélien. Simple et clair.",
        "photo_realiste": "Style photo-réaliste, contexte rural africain, éclairage naturel.",
    }

    is_urgency = style == "schema"
    visual_prompt = _sanitize_visual_prompt(prompt, is_urgency=is_urgency)
    enriched_prompt = (
        f"{visual_prompt}\n\n"
        f"{style_instructions.get(style, style_instructions['illustration'])}\n\n"
        f"{_visual_context_block(category, is_urgency=is_urgency)}\n\n"
        "IMPORTANT: Pas de texte dans l'image. Uniquement des visuels simples, utiles et non choquants."
    )

    try:
        client = _get_media_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_IMAGE_MODEL,
            contents=[enriched_prompt],
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=genai_types.ImageConfig(
                    aspect_ratio=_image_aspect_ratio(style),
                ),
            ),
        )

        image_base64 = None
        mime_type = "image/png"

        for part in getattr(response, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                image_base64 = base64.b64encode(inline_data.data).decode("utf-8")
                mime_type = inline_data.mime_type or "image/png"
                break

        if not image_base64:
            return {
                "success": False,
                "error": "Aucune image générée par le modèle",
                "fallback_description": prompt,
            }

        return {
            "success": True,
            "image_base64": image_base64,
            "mime_type": mime_type,
        }
    except Exception as e:
        print(f"[imageGenerator] Erreur: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_description": prompt,
        }


# ══════════════════════════════════════════════════════
# GÉNÉRATEUR DE VIDÉOS (Veo via REST)
# ══════════════════════════════════════════════════════

async def generate_video(
    prompt: str,
    gemini_api_key: str,
    duration_sec: int = 5,
    is_urgency: bool = False,
    category: str = "agriculture",
) -> dict:
    """Génère une courte vidéo pédagogique via Veo (Gemini uniquement).
    Retourne un fallback texte quand AI_PROVIDER=openai."""
    if not prompt or not prompt.strip():
        return _video_fallback(prompt, duration_sec)

    # ── Routing provider ──────────────────────────────
    if AI_PROVIDER == "openai":
        # OpenAI n'a pas de génération vidéo — fallback texte structuré
        return _video_fallback(prompt, duration_sec, error="Génération vidéo indisponible avec OpenAI — réactiver Gemini (AI_PROVIDER=gemini)")

    duration_sec = _normalize_video_duration(duration_sec)
    aspect_ratio = "9:16" if is_urgency else "16:9"
    visual_prompt = _sanitize_visual_prompt(prompt, is_urgency=is_urgency)
    enriched_prompt = (
        f"Vidéo pédagogique d'urgence de {duration_sec} secondes. {visual_prompt} "
        "Montrer un geste calme, non graphique, démontré par des mains propres ou un mannequin de formation. "
        f"{_visual_context_block('urgence', is_urgency=True)} "
        "Aucun sang, aucune plaie visible, aucun texte à l'écran. Mouvement lent et clair."
        if is_urgency
        else f"Vidéo pédagogique de {duration_sec} secondes. {visual_prompt} "
        f"{_visual_context_block(category, is_urgency=False)} "
        "Montrer une démonstration simple, propre, étape par étape. Aucun texte à l'écran."
    )

    try:
        client = _get_media_client(gemini_api_key)
        operation = await asyncio.to_thread(
            client.models.generate_videos,
            model=VEO_MODEL,
            prompt=enriched_prompt,
            config=genai_types.GenerateVideosConfig(
                duration_seconds=duration_sec,
                aspect_ratio=aspect_ratio,
            ),
        )

        max_wait = 90 if is_urgency else 120
        start_time = time.time()

        while not getattr(operation, "done", False) and (time.time() - start_time) < max_wait:
            await asyncio.sleep(10)
            operation = await asyncio.to_thread(client.operations.get, operation)

        if not getattr(operation, "done", False):
            return _video_fallback(prompt, duration_sec, error="Délai Veo dépassé")

        response = getattr(operation, "response", None)
        generated_videos = getattr(response, "generated_videos", None) or []
        if generated_videos:
            video = generated_videos[0]
            video_bytes = await asyncio.to_thread(client.files.download, file=video.video)
            return {
                "success": True,
                "video_base64": base64.b64encode(video_bytes).decode("utf-8"),
                "video_url": None,
                "mime_type": "video/mp4",
                "duration_sec": duration_sec,
            }

        filtered_reasons = getattr(response, "rai_media_filtered_reasons", None) or []
        if filtered_reasons:
            return _video_fallback(prompt, duration_sec, error=" ; ".join(filtered_reasons))

        return _video_fallback(prompt, duration_sec, error="Aucune vidéo retournée par Veo")

    except Exception as e:
        print(f"[videoGenerator] Erreur: {e}")
        return _video_fallback(prompt, duration_sec, error=str(e))


def _video_fallback(prompt: str, duration_sec: int, error: Optional[str] = None) -> dict:
    return {
        "success": False,
        "fallback": True,
        "video_description": prompt,
        "steps_visuelles": [
            "Étape 1 : Regardez bien la zone concernée",
            "Étape 2 : Préparez le matériel nécessaire",
            "Étape 3 : Suivez les gestes décrits lentement",
        ],
        "duration_sec": duration_sec,
        "error": error or "Génération vidéo non disponible - utilisez les instructions étape par étape",
    }


# ══════════════════════════════════════════════════════
# CONSTRUCTEUR DE RÉPONSE UNIQUE
# ══════════════════════════════════════════════════════

def build_response(
    analysis: dict,
    decision: dict,
    image_result: Optional[dict] = None,
    video_result: Optional[dict] = None,
) -> dict:
    """Construit LA réponse finale unique"""

    message = _build_human_message(analysis, decision)
    actions = _build_actions_list(analysis, decision)

    response = {
        "message": message,
        "diagnostic": {
            "type": analysis["type_probleme"],
            "description": analysis["diagnostic"],
            "gravite": analysis["gravite"],
            "confiance": analysis["confiance"],
            "causes": analysis["causes_probables"],
            "description_visuelle": analysis.get("description_visuelle") or None,
        },
        "actions": actions,
        "image_url": None,
        "image_base64": None,
        "video_url": None,
        "video_base64": None,
        "urgence": decision["mode_urgence"],
        "priorite": decision["priorite"],
        "consulter_expert": decision["transfert_expert"],
        "format": decision["format_reponse"],
    }

    # Image
    if image_result and image_result.get("success"):
        response["image_base64"] = image_result["image_base64"]
        response["image_mime_type"] = image_result.get("mime_type", "image/png")
    elif image_result and image_result.get("fallback_description"):
        response["image_description"] = image_result["fallback_description"]

    # Vidéo
    if video_result and video_result.get("success"):
        response["video_base64"] = video_result.get("video_base64")
        response["video_url"] = video_result.get("video_url")
        response["video_mime_type"] = video_result.get("mime_type", "video/mp4")
        response["video_duration"] = video_result.get("duration_sec")
    elif video_result and video_result.get("fallback"):
        response["video_description"] = video_result.get("video_description")
        response["video_steps"] = video_result.get("steps_visuelles")

    return response


def _build_human_message(analysis: dict, decision: dict) -> str:
    diagnostic = analysis["diagnostic"]
    gravite = analysis["gravite"]
    type_probleme = analysis["type_probleme"]
    confiance = analysis["confiance"]

    # MODE URGENCE
    if decision["mode_urgence"]:
        urgency_header = "🚨 URGENCE - Agissez maintenant !\n\n"
        short_diag = diagnostic[:150] + "..." if len(diagnostic) > 150 else diagnostic
        immediate_steps = "\n".join(
            f"{i+1}. {action}" for i, action in enumerate(analysis.get("actions_immediates", [])[:5])
        )
        emergency_line = f"\n\n📞 Appelez le {EMERGENCY_NUMBERS['samu']} (urgence) ou le {EMERGENCY_NUMBERS['pompiers']} (pompiers)"
        return urgency_header + short_diag + "\n\n" + immediate_steps + emergency_line

    # STANDARD / DÉTAILLÉ
    header = {"agriculture": "🌿 Diagnostic agricole\n\n", "elevage": "🐄 Diagnostic animal\n\n"}.get(type_probleme, "ℹ️ Résultat\n\n")

    visual_part = f"👁️ Ce que j'observe :\n{analysis['description_visuelle']}\n\n" if analysis.get("description_visuelle") else ""

    diag_part = f"📋 {diagnostic}\n"

    confiance_part = ""
    if confiance < 0.5:
        confiance_part = "\n⚠️ Je ne suis pas très sûr de ce diagnostic. Ajoutez une photo plus nette ou plus de détails.\n"
    elif confiance < 0.7:
        confiance_part = "\nℹ️ Ce diagnostic est probable mais mériterait confirmation par un expert.\n"

    causes_part = ""
    if analysis.get("causes_probables"):
        causes_part = "\n🔍 Causes possibles :\n" + "\n".join(f"• {c}" for c in analysis["causes_probables"]) + "\n"

    expert_part = ""
    if analysis.get("consulter_expert") and analysis.get("message_expert"):
        expert_part = f"\n👨‍⚕️ {analysis['message_expert']}\n"

    return header + visual_part + diag_part + confiance_part + causes_part + expert_part


def _build_actions_list(analysis: dict, decision: dict) -> list:
    actions = []
    index = 1

    if decision["mode_urgence"]:
        for action in analysis.get("actions_immediates", []):
            actions.append({"numero": index, "texte": action, "type": "immediat", "priorite": "critique"})
            index += 1
        return actions

    for action in analysis.get("actions_immediates", []):
        actions.append({"numero": index, "texte": action, "type": "immediat", "priorite": "haute"})
        index += 1

    existing = {a["texte"].lower().strip() for a in actions}
    for action in analysis.get("actions_detaillees", []):
        if action.lower().strip() not in existing:
            existing.add(action.lower().strip())
            actions.append({"numero": index, "texte": action, "type": "detaille", "priorite": "moyenne"})
            index += 1

    for action in analysis.get("actions_preventives", []):
        if action.lower().strip() not in existing:
            existing.add(action.lower().strip())
            actions.append({"numero": index, "texte": action, "type": "prevention", "priorite": "conseil"})
            index += 1

    return actions


# ══════════════════════════════════════════════════════
# GEMINI POUR RAG (remplace OpenAI dans generate_llm_answer)
# ══════════════════════════════════════════════════════

async def gemini_llm_answer(
    question: str,
    language: str,
    domain: str,
    knowledge_items: List[Dict[str, Any]],
    conversation_context: Optional[List[Dict[str, str]]] = None,
    focus_context: Optional[Dict[str, Any]] = None,
    photo_analysis: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Remplace generate_llm_answer - utilise Gemini au lieu d'OpenAI"""
    if not knowledge_items:
        return None

    focus_subject_label = ""
    focus_issue_label = ""
    if focus_context:
        if focus_context.get("subject"):
            focus_subject_label = focus_context["subject"].get("label", "")
        if focus_context.get("issue"):
            focus_issue_label = focus_context["issue"].get("label", "")

    context_blocks = []
    for idx, item in enumerate(knowledge_items, start=1):
        context_blocks.append(
            f"FICHE {idx} ({item.get('domain', '')}) - {item.get('title', '')}\n"
            f"Question typique: {item.get('question', '')}\n"
            f"Réponse validée: {item.get('answer', '')}\n"
            f"Mots-clés: {', '.join(item.get('tags', []))}\n"
        )
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "Tu es Songra, l'agent d'assistance IA de Yingr-AI (Yingr Artificial Intelligence). \n"
        "Yingr-AI est une intelligence artificielle LOCALE et SOUVERAINE basée au Burkina Faso. \n"
        "Tu es le lien entre la connaissance validée et les communautés rurales du Burkina Faso. \n\n"
        "RÈGLES STRICTES : \n"
        "- Tu dois répondre UNIQUEMENT avec les fiches ci-dessous. \n"
        "- Si les fiches ne suffisent pas, dis-le clairement. \n"
        "- Langage TRÈS simple, phrases courtes, concret, sans jargon. \n"
        f"- Tu parles comme un {_category_expertise(domain)}. \n"
        "- TOUJOURS recommander de vérifier avec un expert local. \n"
    )

    focus_instruction = ""
    if focus_subject_label or focus_issue_label:
        parts = [l for l in [focus_subject_label, focus_issue_label] if l]
        focus_instruction = f"\nContrainte: reste centré sur : {' / '.join(parts)}.\n"

    conversation_text = ""
    if conversation_context:
        turns = []
        for turn in conversation_context[-6:]:
            role = "Utilisateur" if turn.get("role") == "user" else "Assistant"
            content = (turn.get("content") or "").strip()
            if content:
                turns.append(f"{role}: {content}")
        if turns:
            conversation_text = "\n\nConversation:\n" + "\n".join(turns) + "\n"

    # Diagnostic photo
    photo_section = ""
    if photo_analysis and (photo_analysis.get("disease_detected") or photo_analysis.get("observations")):
        parts = []
        if photo_analysis.get("disease_detected"):
            parts.append(f"Problème : {photo_analysis['disease_detected']}")
        if photo_analysis.get("detected_subject"):
            parts.append(f"Sujet : {photo_analysis['detected_subject']}")
        if photo_analysis.get("observations"):
            parts.append(f"Observations : {photo_analysis['observations']}")
        photo_section = "📸 DIAGNOSTIC PHOTO :\n" + " | ".join(parts) + "\n"

    if photo_section:
        user_prompt = (
            f"Domaine: {domain}. Langue: {language or 'fr'}.\n\n"
            f"{photo_section}\n"
            f"FICHES :\n{context_text}\n\n"
            f"{focus_instruction}{conversation_text}"
            "Commence par 'D'après l'analyse de ta photo :'. Max 15 phrases. Langage SIMPLE."
        )
    else:
        user_prompt = (
            f"Domaine: {domain}. Langue: {language or 'fr'}.\n"
            f"Question: {question}\n\n"
            f"FICHES :\n{context_text}\n\n"
            f"{focus_instruction}{conversation_text}"
            "Structure : diagnostic, recommandations pratiques, quand consulter expert. Max 15 phrases."
        )

    # ── Routing provider ──────────────────────────────
    if AI_PROVIDER == "openai":
        return await _openai_llm_answer(system_prompt, user_prompt)

    # ── Gemini (réactiver via AI_PROVIDER=gemini) ─────
    model = _get_model()
    full_prompt = system_prompt + "\n\n" + user_prompt

    try:
        result = await _generate_content_with_timeout(model, full_prompt)
        return result.text
    except Exception as e:
        print(f"[gemini_llm_answer] Erreur: {e}")
        return None


async def gemini_llm_general_knowledge(
    question: str,
    language: str,
    domain: str,
    conversation_context: Optional[List[Dict[str, str]]] = None,
    photo_analysis: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Remplace generate_llm_answer_with_general_knowledge - Gemini au lieu d'OpenAI"""
    domain_description = {
        "agriculture": "l'agriculture et les cultures",
        "elevage": "l'élevage et l'élevage du bétail",
        "health": "les premiers secours et la sécurité sanitaire",
        "cybersecurity": "la cybersécurité et la sécurité en ligne",
    }.get(domain, domain)

    system_prompt = (
        "Tu es Songra, un assistant rural qui aide les communautés du Burkina Faso. \n"
        f"Spécialité actuelle : {domain_description}. \n"
        f"Tu parles comme un {_category_expertise(domain)}. \n"
        "Réponses SIMPLES, PRATIQUES, HONNÊTES. \n"
        "Tu n'as pas de fiche spécialisée, tu utilises tes connaissances générales. \n"
    )

    conversation_text = ""
    if conversation_context:
        turns = []
        for turn in conversation_context[-6:]:
            role = "Utilisateur" if turn.get("role") == "user" else "Assistant"
            content = (turn.get("content") or "").strip()
            if content:
                turns.append(f"{role}: {content}")
        if turns:
            conversation_text = "\nConversation:\n" + "\n".join(turns) + "\n"

    photo_section = ""
    if photo_analysis and (photo_analysis.get("disease_detected") or photo_analysis.get("observations")):
        parts = []
        if photo_analysis.get("disease_detected"):
            parts.append(f"Problème : {photo_analysis['disease_detected']}")
        if photo_analysis.get("detected_subject"):
            parts.append(f"Sujet : {photo_analysis['detected_subject']}")
        if photo_analysis.get("observations"):
            parts.append(f"Observations : {photo_analysis['observations']}")
        photo_section = "📸 DIAGNOSTIC PHOTO :\n" + " | ".join(parts) + "\n"

    if photo_section:
        user_prompt = (
            f"Domaine: {domain}. Langue: {language or 'fr'}.\n\n"
            f"{photo_section}\n"
            "Commence par 'D'après l'analyse de ta photo :'. Max 15 phrases.\n"
            f"{conversation_text}"
        )
    else:
        user_prompt = (
            f"Langue : {language or 'fr'}. Domaine : {domain}.\n"
            f"Question : {question}\n{conversation_text}\n"
            "Aide cette personne. Conseils concrets numérotés. Max 15 phrases."
        )

    # ── Routing provider ──────────────────────────────
    if AI_PROVIDER == "openai":
        return await _openai_llm_answer(system_prompt, user_prompt)

    # ── Gemini (réactiver via AI_PROVIDER=gemini) ─────
    model = _get_model()
    full_prompt = system_prompt + "\n\n" + user_prompt

    try:
        result = await _generate_content_with_timeout(model, full_prompt)
        return result.text
    except Exception as e:
        print(f"[gemini_llm_general] Erreur: {e}")
        return None
