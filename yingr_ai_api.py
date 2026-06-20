"""
SONGRA AI Rural Agent - Yingr-AI (IA Souveraine du Burkina Faso)
Fichier contenant les routes d'API FastAPI pour la suite d'IA locale Yingr-AI.
"""

from fastapi import APIRouter, HTTPException, Request
import base64
import json
import time

import yingr_ai_services

router = APIRouter(prefix="/api/v3/yingr-ai", tags=["Yingr-AI (IA Souveraine du Burkina Faso)"])


@router.post("/diagnose/agriculture")
async def diagnose_agriculture(request: Request):
    """
    Diagnostic Agricole par Photo (Feuille malade/ravageur)
    Exécuté sur serveur local Yingr-AI (Qwen2-VL-7B-Instruct + RAG)
    """
    start_time = time.time()
    query_text = ""
    image_b64 = None
    
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        try:
            body = await request.json()
            query_text = body.get("text", "")
            image_b64 = body.get("photo_base64")
            if not image_b64 and body.get("photo_base64_list"):
                photo_list = body.get("photo_base64_list")
                if len(photo_list) > 0:
                    image_b64 = photo_list[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON invalide: {str(e)}")
            
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
            query_text = form.get("text", "")
            file_item = form.get("file")
            if file_item:
                file_bytes = await file_item.read()
                image_b64 = base64.b64encode(file_bytes).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Form-data invalide: {str(e)}")
    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type non supporté. Utilisez application/json ou multipart/form-data."
        )
        
    if not image_b64:
        raise HTTPException(
            status_code=400, 
            detail="Une photo de la feuille ou de la culture est obligatoire pour le diagnostic."
        )

    # 2. Étape RAG : Recherche de fiches de bonnes pratiques
    rag_items = yingr_ai_services.retrieve_rag_context(query_text, domain="agriculture")
    rag_text = "\n".join([f"- Fiche {item['title']}: {item['answer']}" for item in rag_items])

    # 3. Construction du prompt Qwen2-VL
    prompt = f"""Tu es SONGRA, expert agronome de terrain au Burkina Faso.
Analyse cette image agricole et réponds obligatoirement en format JSON valide contenant les clés suivantes:
- culture_detected: la culture détectée (ex: Maïs, Tomate, etc.)
- disease_detected: le diagnostic ou la carence détectée (ex: Carence en Azote, Mildiou)
- severity: 'faible', 'moyenne' ou 'critique'
- symptoms: liste de symptômes observés sur l'image
- local_remedies: liste de traitements naturels et bio-locaux accessibles au Sahel
- chemical_treatments: liste de traitements chimiques si nécessaire en dernier recours
- prevention: mesures préventives pour le prochain cycle
- analysis: résumé en français simple pour un producteur local

Voici le contexte de connaissances validées (RAG) à exploiter en priorité:
{rag_text}

Description utilisateur supplémentaire: {query_text}
Réponds UNIQUEMENT avec le JSON valide, sans blabla ni balises Markdown."""

    try:
        # 4. Inférence Yingr-AI (Vision locale)
        raw_response = await yingr_ai_services.run_yingr_ai_inference(prompt, category="agriculture", image_b64=image_b64)
        
        clean_json = raw_response.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()
        
        response_data = json.loads(clean_json)
        
        duration_ms = int((time.time() - start_time) * 1000)
        response_data["_meta"] = {
            "duration_ms": duration_ms,
            "system": "Yingr-AI (Souveraine du Burkina Faso)",
            "model_vision": "Qwen2-VL-7B-Instruct",
            "rag_sources_count": len(rag_items)
        }
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur d'analyse sur le serveur Yingr-AI : {str(e)}"
        )


@router.post("/diagnose/elevage")
async def diagnose_elevage(request: Request):
    """
    Diagnostic Élevage par Photo (Animal malade/lésion)
    Exécuté sur serveur local Yingr-AI (Qwen2-VL-7B-Instruct + RAG)
    """
    start_time = time.time()
    query_text = ""
    image_b64 = None
    
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        try:
            body = await request.json()
            query_text = body.get("text", "")
            image_b64 = body.get("photo_base64")
            if not image_b64 and body.get("photo_base64_list"):
                photo_list = body.get("photo_base64_list")
                if len(photo_list) > 0:
                    image_b64 = photo_list[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON invalide: {str(e)}")
            
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
            query_text = form.get("text", "")
            file_item = form.get("file")
            if file_item:
                file_bytes = await file_item.read()
                image_b64 = base64.b64encode(file_bytes).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Form-data invalide: {str(e)}")
    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type non supporté. Utilisez application/json ou multipart/form-data."
        )
        
    if not image_b64:
        raise HTTPException(
            status_code=400, 
            detail="Une photo de l'animal est obligatoire pour formuler un diagnostic vétérinaire."
        )

    # 1. Étape RAG : Recherche documentaire
    rag_items = yingr_ai_services.retrieve_rag_context(query_text, domain="elevage")
    rag_text = "\n".join([f"- Fiche {item['title']}: {item['answer']}" for item in rag_items])

    # 2. Prompt Vision
    prompt = f"""Tu es SONGRA, vétérinaire conseil pour les élevages ruraux du Sahel.
Analyse cette image et réponds obligatoirement en format JSON valide avec les clés suivantes:
- animal_species: l'espèce de l'animal (ex: Chèvre, Zébu, Poule)
- disease_detected: suspicion de maladie ou blessure (ex: Gale cutanée, Plaie infectée)
- severity: 'faible', 'moyenne' ou 'critique'
- symptoms: liste de symptômes visibles sur la photo
- local_remedies: remèdes traditionnels de soins vétérinaires locaux (huile de neem, cendres, isolation)
- treatment_steps: étapes de traitement simples numérotées
- when_to_call_vet: critères critiques indiquant qu'il faut appeler le vétérinaire de zone
- analysis: résumé en français simple pour l'éleveur

Base documentaire locale RAG :
{rag_text}

Description utilisateur supplémentaire: {query_text}
Réponds UNIQUEMENT avec le JSON valide, pas d'autre texte."""

    try:
        # 3. Inférence Yingr-AI (Vision locale)
        raw_response = await yingr_ai_services.run_yingr_ai_inference(prompt, category="elevage", image_b64=image_b64)
        
        clean_json = raw_response.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()
        
        response_data = json.loads(clean_json)
        
        duration_ms = int((time.time() - start_time) * 1000)
        response_data["_meta"] = {
            "duration_ms": duration_ms,
            "system": "Yingr-AI (Souveraine du Burkina Faso)",
            "model_vision": "Qwen2-VL-7B-Instruct",
            "rag_sources_count": len(rag_items)
        }
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur d'analyse vétérinaire sur le serveur Yingr-AI : {str(e)}"
        )


@router.post("/sos")
async def sos_accident(request: Request):
    """
    SOS Accident Rural (Transcription vocale + Premiers secours)
    Exécuté sur serveur local Yingr-AI (Whisper Large-v3 + RAG + Qwen2.5)
    """
    start_time = time.time()
    query_text = ""
    audio_transcription = None
    
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        try:
            body = await request.json()
            query_text = body.get("text", "")
            audio_b64 = body.get("audio_base64")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                audio_transcription = await yingr_ai_services.transcribe_audio_whisper(audio_bytes, "audio.wav")
                query_text = audio_transcription
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON invalide: {str(e)}")
            
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
            query_text = form.get("text", "")
            audio_item = form.get("audio")
            if audio_item:
                audio_bytes = await audio_item.read()
                audio_transcription = await yingr_ai_services.transcribe_audio_whisper(audio_bytes, audio_item.filename)
                query_text = audio_transcription
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Form-data invalide: {str(e)}")
    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type non supporté. Utilisez application/json ou multipart/form-data."
        )

    if not query_text or not query_text.strip():
        raise HTTPException(
            status_code=400, 
            detail="Veuillez envoyer un message textuel ou un enregistrement audio décrivant la situation."
        )

    # 2. Étape RAG : Recherche de fiches de premiers secours
    rag_items = yingr_ai_services.retrieve_rag_context(query_text, domain="sos_accident")
    rag_text = "\n".join([f"- Fiche {item['title']}: {item['answer']}" for item in rag_items])

    # 3. Prompt de secourisme (posture Sapeur-Pompier)
    prompt = f"""Tu es un secouriste professionnel (sapeur-pompier) au Burkina Faso.
Analyse la description d'accident suivante et réponds en format JSON valide avec ces clés:
- is_emergency: true ou false
- severity: 'critique', 'haute', 'moyenne' ou 'faible'
- emergency_type: nature de l'accident (ex: morsure de serpent, chute, saignement)
- immediate_actions: liste d'actions de premiers secours claires et ordonnées (🚨 gestes de survie prioritaires)
- call_emergency_number: numéros des secours au Burkina Faso (18, 112, 17)
- first_aid_details: explications additionnelles sur ce qu'il ne faut pas faire (éviter les remèdes toxiques)
- analysis: résumé explicatif simple de la situation

Voici les consignes de premiers secours médicales validées (RAG) à intégrer:
{rag_text}

Description de l'accident : {query_text}
Réponds uniquement avec du JSON valide."""

    try:
        # 4. Inférence Textuelle Yingr-AI
        raw_response = await yingr_ai_services.run_yingr_ai_inference(prompt, category="sos")
        
        clean_json = raw_response.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()
        
        response_data = json.loads(clean_json)
        
        if audio_transcription:
            response_data["audio_transcription"] = audio_transcription
            
        duration_ms = int((time.time() - start_time) * 1000)
        response_data["_meta"] = {
            "duration_ms": duration_ms,
            "system": "Yingr-AI (Souveraine du Burkina Faso)",
            "model_transcription": "Whisper Large-v3",
            "model_text": "Qwen2.5-72B-Instruct-AWQ",
            "rag_sources_count": len(rag_items)
        }
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de traitement SOS sur le serveur Yingr-AI : {str(e)}"
        )
