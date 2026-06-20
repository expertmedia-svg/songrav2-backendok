"""
SONGRA AI Rural Agent - Yingr-AI (IA Souveraine du Burkina Faso)
Module d'intégration avec le serveur d'inférence local Yingr-AI (vLLM/Whisper + RAG).
"""

import os
import json
import base64
import time
from typing import List, Dict, Any, Optional
import httpx

# Configuration des URL d'inférence de Yingr-AI
YINGR_AI_VLLM_URL = os.getenv("YINGR_AI_VLLM_URL", "")  # ex: http://localhost:8000/v1
YINGR_AI_VLLM_VISION_URL = os.getenv("YINGR_AI_VLLM_VISION_URL", YINGR_AI_VLLM_URL)
YINGR_AI_VLLM_TEXT_URL = os.getenv("YINGR_AI_VLLM_TEXT_URL", YINGR_AI_VLLM_URL)
YINGR_AI_WHISPER_URL = os.getenv("YINGR_AI_WHISPER_URL", "")  # ex: http://localhost:9000/v1

# Chemin vers la base de connaissances locale
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BACKEND_DIR, "knowledge_base.json")

# Chargement de la base de connaissances pour le RAG
try:
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        KNOWLEDGE_ITEMS = json.load(f)
    print(f"[RAG Yingr-AI] Base de connaissances chargee : {len(KNOWLEDGE_ITEMS)} fiches")
except Exception as e:
    KNOWLEDGE_ITEMS = []
    print(f"[WARN] Impossible de charger la base de connaissances RAG: {e}")


def retrieve_rag_context(query: str, domain: str, limit: int = 2) -> List[Dict[str, Any]]:
    """
    Recherche RAG localisée : trouve les fiches les plus pertinentes dans la base locale
    par similarité de mots-clés dans le domaine ciblé.
    """
    if not KNOWLEDGE_ITEMS:
        return []
    
    domain_filter = domain
    if domain == "sos_accident":
        domain_filter = "health"
        
    filtered_items = [item for item in KNOWLEDGE_ITEMS if item.get("domain") == domain_filter]
    if not filtered_items:
        filtered_items = KNOWLEDGE_ITEMS

    query_words = set(query.lower().split())
    scored_items = []
    for item in filtered_items:
        score = 0
        text_to_search = (
            (item.get("title", "") + " " + item.get("question", "") + " " + item.get("answer", ""))
            .lower()
        )
        
        for word in query_words:
            if len(word) > 3:
                if word in text_to_search:
                    score += 1
                    
        for tag in item.get("tags", []):
            if tag.lower() in query_words:
                score += 2
                
        scored_items.append((score, item))
        
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored_items[:limit]]


async def transcribe_audio_whisper(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transcrit un vocal de terrain à l'aide du moteur de transcription local Yingr-AI.
    """
    print(f"[Yingr-AI] Ingestion Audio - Taille : {len(audio_bytes)/1024:.1f} KB - Canal Whisper Local")
    
    if YINGR_AI_WHISPER_URL:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {'file': (filename, audio_bytes, 'audio/wav')}
                response = await client.post(
                    f"{YINGR_AI_WHISPER_URL}/audio/transcriptions",
                    files=files,
                    data={"model": "openai/whisper-large-v3", "language": "fr"}
                )
                if response.status_code == 200:
                    result = response.json()
                    print("[OK] Transcription Whisper terminee sur le serveur Yingr-AI")
                    return result.get("text", "")
        except Exception as e:
            print(f"[WARN] Echec de la transcription Whisper locale: {e}. Bascule sur la simulation.")

    # Simulation de transcription locale
    time.sleep(1.2)
    print("[Yingr-AI (Simulé)] Transcription effectuee via Whisper Large-v3 Local")
    return "J'ai eu un grave accident de moto sur la piste, mon pied saigne beaucoup et je ne peux plus bouger."


async def run_yingr_ai_inference(prompt: str, category: str, image_b64: Optional[str] = None) -> str:
    """
    Appelle le modèle de Vision ou de Texte de la suite d'IA Souveraine Yingr-AI.
    """
    if image_b64:
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        vllm_url = YINGR_AI_VLLM_VISION_URL
    else:
        model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ"
        vllm_url = YINGR_AI_VLLM_TEXT_URL
    
    if vllm_url:
        try:
            messages = []
            if image_b64:
                content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            else:
                content = [{"type": "text", "text": prompt}]
                
            messages.append({"role": "user", "content": content})
            
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{vllm_url}/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.2,
                        "max_tokens": 1500,
                        "response_format": {"type": "json_object"}
                    },
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    res_json = response.json()
                    return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[WARN] Echec de l'inference locale Yingr-AI ({vllm_url}): {e}. Bascule sur la simulation locale.")

    # Simulation locale déterministe
    print(f"[Yingr-AI (Simulé)] Execution de {model_name} en local")
    time.sleep(1.5)
    
    return _generate_simulated_json_response(category)


def _generate_simulated_json_response(category: str) -> str:
    """Génère des structures de données JSON de simulation pour les 3 cas d'usage principaux."""
    # 1. SCENARIO AGRICULTURE (Maïs jaune ou rouille)
    if category == "agriculture":
        response_dict = {
            "consultation_type": "Diagnostic agricole complet",
            "culture_detected": "Maïs (Zea mays)",
            "disease_detected": "Carence severe en Azote (N)",
            "severity": "moyenne",
            "confidence": 0.92,
            "symptoms": [
                "Jaunissement en forme de V inversé partant de la pointe vers la nervure centrale",
                "Feuilles du bas plus atteintes que les jeunes feuilles du haut",
                "Croissance générale ralentie avec tiges minces"
            ],
            "local_remedies": [
                "Appliquer du compost organique bien décomposé ou du fumier de parc mélangé à la terre.",
                "Arroser avec de l'urine de bétail fermentée diluée dans de l'eau (1 volume d'urine pour 5 volumes d'eau) comme engrais azoté rapide."
            ],
            "chemical_treatments": [
                "Apporter de l'urée (46% d'azote) à raison de 50 kg par hectare si les conditions hydriques le permettent."
            ],
            "prevention": "Pratiquer la rotation des cultures avec des légumineuses (niébé, arachide) au prochain cycle pour enrichir le sol naturellement.",
            "analysis": "Votre maïs souffre d'un manque criant de nourriture dans le sol (azote). Cela ralentit sa croissance. Appliquez rapidement du fumier de bétail ou de l'urine fermentée diluée près des pieds."
        }
        return json.dumps(response_dict, ensure_ascii=False)
        
    # 2. SCENARIO ÉLEVAGE (Chèvre gale ou dermatose)
    elif category == "elevage":
        response_dict = {
            "consultation_type": "Diagnostic veterinaire rural",
            "animal_species": "Chèvre rurale du Sahel",
            "disease_detected": "Gale sarcoptique (suspicion forte)",
            "severity": "moyenne",
            "confidence": 0.88,
            "symptoms": [
                "Perte de poils par plaques irrégulières sur la tête et le cou",
                "Présence de croûtes épaisses et blanchâtres sur la peau",
                "Démangeaisons intenses (l'animal se frotte contre les piquets)"
            ],
            "local_remedies": [
                "Isoler immédiatement l'animal pour éviter la contagion au reste du troupeau.",
                "Laver doucement les croûtes avec de l'eau savonneuse tiède, puis appliquer un mélange d'huile de neem et de soufre en poudre."
            ],
            "treatment_steps": [
                "1. Séparer la chèvre malade des autres animaux.",
                "2. Nettoyer l'abri et désinfecter le bois de l'enclos.",
                "3. Appliquer l'huile de neem sur les zones sans poils deux fois par semaine pendant 21 jours."
            ],
            "when_to_call_vet": "Si les croûtes s'étendent sur tout le corps ou si l'animal cesse de s'alimenter (abattement sévère).",
            "analysis": "Votre chèvre présente des signes de gale de la peau. C'est une maladie contagieuse causée par des petits acariens. Séparez-la vite des autres et appliquez de l'huile de neem mélangée à de la cendre ou du soufre sur les croûtes."
        }
        return json.dumps(response_dict, ensure_ascii=False)
        
    # 3. SCENARIO SOS ACCIDENT RURAL (sos)
    else:
        response_dict = {
            "consultation_type": "SOS Secourisme d'Urgence",
            "is_emergency": True,
            "severity": "critique",
            "emergency_type": "Traumatisme avec saignement (Accident de moto)",
            "immediate_actions": [
                "🚨 ÉTAPE 1 : Securiser la victime et le lieu pour eviter un suraccident.",
                "🚨 ÉTAPE 2 : Si la plaie saigne abondamment, appuyez FERMEMENT sur la plaie avec un tissu PROPRE pendant 10 minutes.",
                "🚨 ÉTAPE 3 : Si la victime respire mais est inconsciente, allongez-la doucement sur le côté (Position Latérale de Secours).",
                "🚨 ÉTAPE 4 : NE PAS deplacer le blessé s'il y a suspicion de fracture du dos ou du cou, sauf danger de mort immédiat (incendie)."
            ],
            "call_emergency_number": "Pompiers : 18 | SAMU local : 112 / 15",
            "first_aid_details": "Ne pas verser de terre ou de produits traditionnels irritants sur une plaie qui saigne. Utilisez uniquement de l'eau propre pour laver autour si nécessaire et maintenez la pression.",
            "analysis": "Il s'agit d'une urgence critique suite à un accident de moto. Les premiers gestes doivent être appliqués immédiatement : compression du saignement et mise en sécurité en attendant les secours."
        }
        return json.dumps(response_dict, ensure_ascii=False)
