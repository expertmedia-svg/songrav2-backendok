"""
v2_services.py - Pipeline v2 unifié (porté de backend-node)
Tout fonctionne avec Gemini uniquement (pas d'OpenAI)

Modules :
- analyse unifiée Gemini (texte + images)
- moteur de décision (image? vidéo? urgence?)
- générateur d'images (Gemini Imagen)
- générateur de vidéos (Veo via REST)
- constructeur de réponse unique
- analyse entrepreneuriale (Entreprendre)
"""

import base64
import json
import re
import time
import asyncio
import httpx
from typing import Optional, List, Dict, Any
import google.generativeai as genai

# ══════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash"  # pour la génération d'images
VEO_MODEL = "veo-2.0-generate-001"
VEO_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MAX_PHOTOS = 3
GEMINI_TIMEOUT = 60  # secondes
CACHE_TTL = 300  # 5 minutes

EMERGENCY_NUMBERS = {"pompiers": "18", "police": "17", "samu": "112"}

# Cache simple en mémoire
_analysis_cache: Dict[str, Dict] = {}


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

CONTEXTE TERRAIN :
- Climat chaud et sec, saison des pluies courte
- Marchés locaux pour les produits de base
- Centres de santé parfois à des heures de route
- Eau propre pas toujours disponible
- Réseau mobile faible - réponses compactes nécessaires"""

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
- En cas d'urgence vitale (gravite = "critique"), les actions_immediates doivent être des gestes de survie numérotés"""


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
- Retourne UNIQUEMENT le JSON, pas de texte avant ni après"""


# ══════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════

def _get_model(model_name: str = GEMINI_MODEL):
    """Obtenir un modèle Gemini configuré"""
    return genai.GenerativeModel(model_name)


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
# SERVICE D'ANALYSE GEMINI UNIFIÉ
# ══════════════════════════════════════════════════════

async def gemini_analyze(
    text: str = "",
    images_b64: Optional[List[str]] = None,
    category: str = "agriculture",
) -> dict:
    """Analyse unifiée via Gemini : texte + images → JSON structuré"""
    images_b64 = images_b64 or []
    has_image = len(images_b64) > 0
    has_text = bool(text.strip())

    if not has_text and not has_image:
        raise ValueError("Veuillez fournir au moins du texte ou une image")

    # Cache (texte seul)
    if not has_image and has_text:
        cache_key = _get_cache_key(text, category, False)
        cached = _analysis_cache.get(cache_key)
        if cached and (time.time() - cached["ts"]) < CACHE_TTL:
            return {**cached["data"], "from_cache": True}

    model = _get_model()

    context_hint = ""
    if category == "elevage":
        context_hint = "\nCONTEXTE : problème lié à un animal d'élevage au Burkina Faso."
    elif category in ("urgence", "sos_accident"):
        context_hint = "\nCONTEXTE : situation d'urgence / accident / blessure humaine. PRIORITÉ AUX GESTES QUI SAUVENT LA VIE."
    else:
        context_hint = "\nCONTEXTE : problème agricole (culture, parcelle, sol) au Burkina Faso."

    user_message = f"\nDescription de l'utilisateur : {text}" if has_text else "\n(Pas de description textuelle - analyse basée sur l'image uniquement)"
    full_prompt = ANALYSIS_PROMPT + context_hint + user_message

    content_parts = [full_prompt]
    for img_b64 in images_b64[:MAX_PHOTOS]:
        content_parts.append({
            "mime_type": "image/jpeg",
            "data": img_b64,
        })

    result = await asyncio.to_thread(model.generate_content, content_parts)
    response_text = result.text
    raw_json = _parse_gemini_json(response_text)
    analysis = _validate_analysis(raw_json)

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
    """Analyse entrepreneuriale terrain via Gemini"""
    images_b64 = images_b64 or []
    has_image = len(images_b64) > 0
    has_text = bool(text.strip())

    if not has_text and not has_image:
        raise ValueError("Envoyez une photo de votre terrain ou décrivez-le")

    model = _get_model()

    context_hint = ""
    if category == "elevage":
        context_hint = "\nCONTEXTE : terrain destiné à l'élevage au Burkina Faso. Propose des projets d'élevage ET de culture mixte."
    else:
        context_hint = "\nCONTEXTE : terrain agricole au Burkina Faso. Propose des projets de culture ET éventuellement d'élevage complémentaire."

    user_message = f"\nDescription de l'utilisateur : {text}" if has_text else "\n(Pas de description textuelle - analyse basée sur l'image uniquement)"
    full_prompt = ENTREPRENDRE_PROMPT + context_hint + user_message

    content_parts = [full_prompt]
    for img_b64 in images_b64[:MAX_PHOTOS]:
        content_parts.append({
            "mime_type": "image/jpeg",
            "data": img_b64,
        })

    result = await asyncio.to_thread(model.generate_content, content_parts)
    response_text = result.text
    raw_json = _parse_gemini_json(response_text)
    return _validate_entrepreneurship(raw_json)


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
        decision["prompt_image"] = f"Illustration simple et claire montrant le geste de premier secours pour : {analysis['diagnostic']}. Style schématique, couleurs vives, compréhensible sans savoir lire. Contexte africain rural."
        decision["generer_video"] = True
        actions_text = ". ".join(analysis.get("actions_immediates", []))
        decision["prompt_video"] = f"Vidéo courte de premiers secours (5-8 secondes) montrant les gestes d'urgence pour : {analysis['diagnostic']}. Actions : {actions_text}. Style simple, contexte : village africain."
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

    # TOUJOURS générer image + vidéo pédagogique
    decision["generer_image"] = True
    decision["generer_video"] = True

    if type_probleme == "agriculture":
        decision["prompt_image"] = f"Illustration pédagogique agricole montrant : {analysis['diagnostic']}. Montrer les symptômes sur la plante et le traitement recommandé. Style dessin simple, adapté pour des agriculteurs au Burkina Faso."
        actions_text = ". ".join(analysis.get("actions_detaillees", [])[:3])
        decision["prompt_video"] = f"Vidéo pédagogique courte (5-10 secondes) montrant les gestes techniques pour traiter : {analysis['diagnostic']}. Étapes : {actions_text}. Contexte : champ en Afrique sahélienne."
    elif type_probleme == "elevage":
        decision["prompt_image"] = f"Illustration vétérinaire simple montrant : {analysis['diagnostic']}. Montrer les signes à observer sur l'animal et les soins de base. Style schématique, adapté éleveurs ruraux Burkina Faso."
        actions_text = ". ".join(analysis.get("actions_detaillees", [])[:3])
        decision["prompt_video"] = f"Vidéo pédagogique courte (5-10 secondes) montrant les soins de base pour : {analysis['diagnostic']}. Étapes : {actions_text}. Contexte : élevage rural Burkina Faso."
    else:
        decision["prompt_image"] = f"Illustration explicative simple pour : {analysis['diagnostic']}. Style clair, compréhensible par tous."
        actions_text = ". ".join(analysis.get("actions_detaillees", [])[:3])
        decision["prompt_video"] = f"Vidéo pédagogique courte (5-10 secondes) pour : {analysis['diagnostic']}. Étapes : {actions_text}."

    return decision


# ══════════════════════════════════════════════════════
# GÉNÉRATEUR D'IMAGES (Gemini)
# ══════════════════════════════════════════════════════

async def generate_image(prompt: str, style: str = "illustration") -> dict:
    """Génère une image explicative via Gemini"""
    if not prompt or not prompt.strip():
        return {"success": False, "error": "Prompt image requis"}

    style_instructions = {
        "schema": "Style schématique simple, fond blanc, traits noirs épais, couleurs vives de base (rouge, vert, jaune). Compréhensible sans savoir lire.",
        "illustration": "Style illustration pédagogique, couleurs chaudes, personnages africains, paysage sahélien. Simple et clair.",
        "photo_realiste": "Style photo-réaliste, contexte rural africain, éclairage naturel.",
    }

    enriched_prompt = f"{prompt}\n\n{style_instructions.get(style, style_instructions['illustration'])}\n\nIMPORTANT: Pas de texte dans l'image. Uniquement des visuels."

    try:
        model = _get_model(GEMINI_IMAGE_MODEL)

        result = await asyncio.to_thread(
            model.generate_content,
            {
                "contents": [{"role": "user", "parts": [{"text": enriched_prompt}]}],
                "generation_config": {"response_modalities": ["TEXT", "IMAGE"]},
            },
        )

        response = result
        image_base64 = None
        mime_type = "image/png"

        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts or []
            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    image_base64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                    mime_type = part.inline_data.mime_type or "image/png"
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

async def generate_video(prompt: str, gemini_api_key: str, duration_sec: int = 5, is_urgency: bool = False) -> dict:
    """Génère une courte vidéo pédagogique via Veo"""
    if not prompt or not prompt.strip():
        return _video_fallback(prompt, duration_sec)

    duration_sec = max(3, min(10, duration_sec))

    enriched_prompt = (
        f"Vidéo d'urgence de {duration_sec} secondes. {prompt} Mouvements lents et clairs, gestes de premiers secours visibles. Pas de texte. Contexte : village africain."
        if is_urgency
        else f"Vidéo pédagogique de {duration_sec} secondes. {prompt} Gestes simples et lents, faciles à reproduire. Pas de texte. Contexte rural Burkina Faso."
    )

    try:
        start_url = f"{VEO_API_BASE}/models/{VEO_MODEL}:predictLongRunning?key={gemini_api_key}"

        async with httpx.AsyncClient(timeout=120) as client:
            start_res = await client.post(
                start_url,
                json={
                    "instances": [{"prompt": enriched_prompt}],
                    "parameters": {"sampleCount": 1},
                },
            )

            if start_res.status_code != 200:
                raise ValueError(f"Veo API {start_res.status_code}: {start_res.text[:200]}")

            operation = start_res.json()
            op_name = operation.get("name")
            if not op_name:
                raise ValueError("Veo : pas de nom d'opération retourné")

            # Polling (max ~90s avec backoff)
            max_wait = 60 if is_urgency else 90
            start_time = time.time()
            poll_delay = 5

            while time.time() - start_time < max_wait:
                await asyncio.sleep(poll_delay)
                poll_delay = min(poll_delay * 1.5, 15)

                poll_url = f"{VEO_API_BASE}/{op_name}?key={gemini_api_key}"
                poll_res = await client.get(poll_url)
                if poll_res.status_code != 200:
                    continue

                poll_data = poll_res.json()
                if not poll_data.get("done"):
                    continue

                predictions = (poll_data.get("response") or {}).get("predictions", [])
                for pred in predictions:
                    if pred.get("bytesBase64Encoded"):
                        return {
                            "success": True,
                            "video_base64": pred["bytesBase64Encoded"],
                            "video_url": None,
                            "mime_type": pred.get("mimeType", "video/mp4"),
                            "duration_sec": duration_sec,
                        }
                    if pred.get("videoUri"):
                        return {
                            "success": True,
                            "video_base64": None,
                            "video_url": pred["videoUri"],
                            "mime_type": "video/mp4",
                            "duration_sec": duration_sec,
                        }

                return _video_fallback(prompt, duration_sec)

            return _video_fallback(prompt, duration_sec)

    except Exception as e:
        print(f"[videoGenerator] Erreur: {e}")
        return _video_fallback(prompt, duration_sec)


def _video_fallback(prompt: str, duration_sec: int) -> dict:
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
        "error": "Génération vidéo non disponible - utilisez les instructions étape par étape",
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
            actions.append({"numero": index, "texte": action, "type": "preventif", "priorite": "normale"})
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

    model = _get_model()
    full_prompt = system_prompt + "\n\n" + user_prompt

    try:
        result = await asyncio.to_thread(model.generate_content, full_prompt)
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

    model = _get_model()
    full_prompt = system_prompt + "\n\n" + user_prompt

    try:
        result = await asyncio.to_thread(model.generate_content, full_prompt)
        return result.text
    except Exception as e:
        print(f"[gemini_llm_general] Erreur: {e}")
        return None
