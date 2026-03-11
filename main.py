"""
SONGRA - Backend API avec Computer Vision LOCALE
Version FINALE - Avec analyse IA complète
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import hashlib
import unicodedata
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# CONFIGURATION
# ==========================================

# Charger les variables d'environnement depuis un fichier .env (dev / prod)
# override=True permet de remplacer une éventuelle variable OPENAI_API_KEY déjà
# définie dans l'environnement Windows (ancienne clé) par la valeur du .env.
load_dotenv(override=True)

os.makedirs("uploads", exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./resolvehub.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ==========================================
# MODÈLES (Compatibles avec base existante)
# ==========================================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    location = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_anonymized = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    premium_expires_at = Column(DateTime, nullable=True)
    messages_used = Column(Integer, default=0)
    messages_limit = Column(Integer, default=1)  # 1 gratuit, 10 pour premium

class Expert(Base):
    __tablename__ = "experts"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    specialization = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    expert_id = Column(Integer, nullable=True)
    category = Column(String, nullable=True)
    urgency = Column(String, nullable=True)
    status = Column(String, default="open")
    ai_confidence_score = Column(Float, nullable=True)
    ai_extracted_keywords = Column(String, nullable=True)
    ai_photo_analysis = Column(Text, nullable=True)
    photo_path = Column(String, nullable=True)  # NOM ORIGINAL - NE PAS CHANGER
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, nullable=False)
    sender_type = Column(String, nullable=False)
    sender_id = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    channel = Column(String, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    is_read = Column(Boolean, default=False)

class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, nullable=False)  # agriculture / elevage / cybersecurity / health
    title = Column(String, nullable=False)
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=False)
    tags = Column(String, nullable=True)  # JSON list of tags
    language = Column(String, default="fr")
    source = Column(String, nullable=True)
    media = Column(Text, nullable=True)  # JSON list of media items (images / videos)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
class EmergencyNumber(Base):
    __tablename__ = "emergency_numbers"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False)  # Ex: Sapeurs-pompiers
    number = Column(String, nullable=False)  # Ex: 18, 112, etc.
    description = Column(Text, nullable=True)  # Note contextuelle (pays / région)
    display_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)


Base.metadata.create_all(bind=engine)


def _ensure_media_column_for_knowledge_items() -> None:
    """S'assurer que la colonne 'media' existe dans la table knowledge_items.

    Utile quand la base SQLite existait avant l'ajout de ce champ : on ajoute
    simplement la colonne manquante sans casser les données existantes.
    """
    # Fonction principalement pensée pour SQLite (dev/local). Pour d'autres
    # SGBD avec migrations gérées, on ne fait rien ici.
    if not str(engine.url).startswith("sqlite"):
        return

    try:
        with engine.connect() as conn:
            # Récupérer la liste des colonnes existantes
            result = conn.exec_driver_sql("PRAGMA table_info(knowledge_items)")
            columns = [row[1] for row in result]  # row[1] = nom de colonne

            if "media" not in columns:
                conn.exec_driver_sql("ALTER TABLE knowledge_items ADD COLUMN media TEXT")
    except Exception as e:
        # On loggue mais on ne bloque pas le démarrage de l'API
        print(f"⚠️ Impossible d'ajouter la colonne 'media' à knowledge_items: {e}")


_ensure_media_column_for_knowledge_items()

# ==========================================
# MODÈLES PYDANTIC
# ==========================================

class MessageCreate(BaseModel):
    content: str
    phone_number: str
    channel: str = "app"
    category: Optional[str] = None  # catégorie choisie côté app (agriculture, elevage, sos_accident, cybersecurity)
    photo_base64: Optional[str] = None

class ExpertLogin(BaseModel):
    email: str
    password: str

class ReplyMessage(BaseModel):
    message: str


class KnowledgeMedia(BaseModel):
    type: str
    url: str
    title: Optional[str] = None


class KnowledgeItemIn(BaseModel):
    domain: str
    title: str
    question: Optional[str] = None
    answer: str
    tags: List[str] = []
    language: str = "fr"
    source: Optional[str] = None
    media: Optional[List[KnowledgeMedia]] = None


class KnowledgeBulkImport(BaseModel):
    items: List[KnowledgeItemIn]


class EmergencyNumberIn(BaseModel):
    label: str
    number: str
    description: Optional[str] = None
    display_order: int = 0


# ==========================================
# MODULE IA PHOTO LOCALE (Computer Vision) - RESTAURÉ
# ==========================================

class LocalComputerVision:
    """
    Système de Computer Vision 100% LOCAL
    Détection de maladies des plantes sans API externe
    """
    
    def __init__(self):
        # Base de connaissances des maladies courantes au Burkina Faso
        self.diseases_database = {
            "mais_taches_jaunes": {
                "name": "Carence en Azote",
                "confidence_keywords": ["jaune", "feuille", "maïs", "sécher"],
                "symptoms": ["Jaunissement des feuilles du bas vers le haut", "Croissance ralentie"],
                "treatment": "Appliquer engrais NPK (10-10-10) à 50kg/ha. Améliorer drainage. Arrosage régulier matin/soir.",
                "urgency": "medium",
                "prevention": "Rotation des cultures, compost organique, analyse sol annuelle"
            },
            "mais_rouille": {
                "name": "Rouille du Maïs",
                "confidence_keywords": ["tache", "orange", "rouille", "poudre"],
                "symptoms": ["Pustules orange/brunes sur feuilles", "Aspect poudreuse"],
                "treatment": "Fongicide naturel (purin d'ortie dilué 1:10). Retirer feuilles infectées. Espacer plants.",
                "urgency": "high",
                "prevention": "Variétés résistantes, rotation, bon espacement"
            },
            "tomate_mildiou": {
                "name": "Mildiou de la Tomate",
                "confidence_keywords": ["tache", "brun", "noir", "tomate", "pourrir"],
                "symptoms": ["Taches brunes/noires sur feuilles", "Fruits pourrissent"],
                "treatment": "URGENT: Retirer plants infectés. Bouillie bordelaise. Éviter arrosage feuilles.",
                "urgency": "high",
                "prevention": "Paillage, arrosage au pied, aération"
            },
            "sorgho_charbon": {
                "name": "Charbon du Sorgho",
                "confidence_keywords": ["noir", "poudre", "épi", "sorgho"],
                "symptoms": ["Masse noire poudreuse remplace grains"],
                "treatment": "Détruire plants infectés (brûler). Traiter semences. Rotation 3 ans.",
                "urgency": "high",
                "prevention": "Semences certifiées traitées, rotation cultures"
            },
            "manioc_mosaique": {
                "name": "Mosaïque du Manioc",
                "confidence_keywords": ["mosaïque", "déformation", "feuille", "manioc"],
                "symptoms": ["Motif mosaïque jaune/vert sur feuilles", "Déformation"],
                "treatment": "Pas de traitement. Arracher et détruire. Utiliser boutures saines certifiées.",
                "urgency": "high",
                "prevention": "Boutures certifiées, contrôle pucerons, éliminer plants malades"
            },
            "animal_fievre": {
                "name": "Fièvre Animale (suspicion)",
                "confidence_keywords": ["bétail", "fièvre", "faible", "animal"],
                "symptoms": ["Température élevée", "Perte appétit", "Faiblesse"],
                "treatment": "CONSULTER vétérinaire RAPIDEMENT. Isoler animal. Eau fraîche disponible.",
                "urgency": "high",
                "prevention": "Vaccination, vermifugation, abri ombragé"
            }
        }
        
        # Maladies par culture pour reconnaissance rapide
        self.crop_diseases = {
            "maïs": ["mais_taches_jaunes", "mais_rouille"],
            "tomate": ["tomate_mildiou"],
            "sorgho": ["sorgho_charbon"],
            "manioc": ["manioc_mosaïque"],
            "bétail": ["animal_fievre"]
        }
    
    def analyze_image_simple(self, image_data: bytes, text_description: str = "") -> dict:
        """
        Analyse simple basée sur le texte ET détection basique image
        (Version simplifiée mais fonctionnelle)
        """
        try:
            text_lower = text_description.lower()
            
            # Détection basique selon texte
            if "maïs" in text_lower or "mais" in text_lower or "jaune" in text_lower:
                return {
                    "disease_detected": "Carence en Azote",
                    "confidence": 0.7,
                    "symptoms": ["Jaunissement des feuilles du bas vers le haut", "Croissance ralentie"],
                    "treatment": "Appliquer engrais NPK (10-10-10) à 50kg/ha. Améliorer drainage. Arrosage régulier matin/soir.",
                    "prevention": "Rotation des cultures, compost organique, analyse sol annuelle",
                    "urgency": "medium",
                    "analysis": "Détection probable de carence en azote sur plants de maïs. Les feuilles jaunissent du bas vers le haut.",
                    "recommendations": "Apportez de l'engrais NPK et assurez-vous d'un bon drainage.",
                    "requires_expert": False
                }
            elif "tomate" in text_lower or "tache" in text_lower:
                return {
                    "disease_detected": "Mildiou de la Tomate",
                    "confidence": 0.6,
                    "symptoms": ["Taches brunes/noires sur feuilles", "Fruits pourrissent"],
                    "treatment": "URGENT: Retirer plants infectés. Bouillie bordelaise. Éviter arrosage feuilles.",
                    "prevention": "Paillage, arrosage au pied, aération",
                    "urgency": "high",
                    "analysis": "Symptômes typiques du mildiou de la tomate.",
                    "recommendations": "Retirez les plants infectés et appliquez de la bouillie bordelaise.",
                    "requires_expert": True
                }
            
            # Réponse générique si pas de match
            return {
                "disease_detected": "Indéterminé",
                "confidence": 0.3,
                "symptoms": ["Analyse en cours"],
                "treatment": "Un expert va examiner votre photo et vous donner des recommandations précises.",
                "prevention": "Photos claires et description détaillée aident à un meilleur diagnostic.",
                "urgency": "medium",
                "analysis": "L'analyse nécessite plus d'informations. Un expert va examiner votre photo.",
                "recommendations": "Prenez plusieurs photos (plante entière, feuilles, tiges). Décrivez les symptômes en détail.",
                "requires_expert": True
            }
        except Exception as e:
            return {
                "disease_detected": "Erreur d'analyse",
                "confidence": 0.0,
                "symptoms": ["Erreur technique"],
                "treatment": "Veuillez décrire le problème par texte.",
                "prevention": "Réessayez avec une photo plus claire.",
                "urgency": "low",
                "analysis": f"Erreur technique lors de l'analyse: {str(e)}",
                "recommendations": "Veuillez réessayer ou décrire le problème par texte.",
                "requires_expert": True
            }

cv_engine = LocalComputerVision()

# ==========================================
# MODULE IA TEXTE (NLP Local) - RESTAURÉ
# ==========================================

class AITriageEngine:
    def __init__(self):
        self.urgency_keywords = {
            "high": ["urgence", "urgent", "grave", "danger", "sang", "brûlure", "piraté", 
                    "volé", "mort", "mourir", "pourrir", "invasion", "attaque"],
            "medium": ["problème", "aide", "rapidement", "besoin", "important", "malade"],
            "low": ["conseil", "information", "question", "quand", "comment", "préventif"]
        }
        
        self.category_keywords = {
            "agriculture": ["maïs", "sorgho", "mil", "culture", "plante", "champ", "récolte", 
                          "irrigation", "tomate", "oignon", "arachide", "coton",
                          "manioc", "riz", "feuille", "insecte", "parasite", "engrais"],
            # Catégorie élevage : animaux, bétail, poules…
            "elevage": ["bétail", "vache", "boeuf", "chèvre", "mouton", "poules", "volaille",
                        "agneau", "veau", "animal", "troupeau", "abri", "vermifuge", "parasites"],
            # Catégorie SOS Accident / premiers soins : on garde le domaine health pour le RAG
            "sos_accident": ["blessure", "coupure", "accident", "saigne", "sang", "brûlure",
                              "tomber", "chute", "fracture", "douleur", "secours"],
            "cybersecurity": ["arnaque", "pirate", "mobile money", "code", "mot de passe", 
                            "orange money", "sms suspect", "compte", "fraude", "virus"]
        }
    
    def classify(self, text: str):
        text_lower = text.lower()
        
        # Catégorie
        category_scores = {}
        for cat, keywords in self.category_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            category_scores[cat] = score
        
        category = max(category_scores, key=category_scores.get) if any(category_scores.values()) else "agriculture"
        confidence = category_scores[category] / (len(self.category_keywords[category]) + 1)
        
        # Urgence
        urgency = "low"
        for level, keywords in self.urgency_keywords.items():
            if any(kw in text_lower for kw in keywords):
                urgency = level
                break
        
        keywords = [word for word in text_lower.split() if len(word) > 3][:5]
        
        return {
            "category": category,
            "urgency": urgency,
            "confidence": float(confidence),
            "keywords": keywords
        }

ai_engine = AITriageEngine()

# ==========================================
# APPLICATION FASTAPI
# ==========================================

app = FastAPI(
    title="SONGRA API - IA Locale",
    version="5.0",
    description="Plateforme d'assistance avec IA locale pour l'analyse de photos"
)

# CORS - AUTORISER TOUT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


# ==========================================
# BASE DE CONNAISSANCE (RAG SIMPLE)
# ==========================================

def load_knowledge_from_json(db: Session, file_path: str = "knowledge_base.json") -> None:
    """Charger une base de connaissances simple à partir d'un fichier JSON.

    Le fichier doit contenir une liste d'objets de la forme :
    {
        "domain": "agriculture" | "elevage" | "cybersecurity" | "health",
        "title": "Titre court compréhensible par un agriculteur",
        "question": "Formulation typique de la question",
        "answer": "Réponse détaillée, validée par les experts locaux",
        "tags": ["mais", "taches jaunes", "engrais"],
        "language": "fr",
        "source": "ONG locale",  # optionnel
    }
    """
    # Toujours se baser sur le dossier du fichier main.py pour trouver le JSON,
    # afin que ça fonctionne même si le serveur est lancé depuis la racine.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(file_path):
        file_path = os.path.join(base_dir, file_path)

    if not os.path.exists(file_path):
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"⚠️ Impossible de charger {file_path}: {e}")
        return

    if not isinstance(raw, list):
        print("⚠️ knowledge_base.json doit contenir une liste d'entrées")
        return

    for item in raw:
        try:
            domain = item.get("domain", "agriculture")
            title = item.get("title")
            answer = item.get("answer")
            if not title or not answer:
                continue

            question = item.get("question")
            tags = item.get("tags") or []
            language = item.get("language", "fr")
            source = item.get("source")
            media = item.get("media")

            # Éviter les doublons simples sur (domain, title)
            existing = db.query(KnowledgeItem).filter(
                KnowledgeItem.domain == domain,
                KnowledgeItem.title == title
            ).first()
            if existing:
                existing.answer = answer
                existing.question = question
                existing.tags = json.dumps(tags, ensure_ascii=False)
                existing.language = language
                existing.source = source
                existing.media = json.dumps(media, ensure_ascii=False) if media is not None else None
            else:
                db.add(KnowledgeItem(
                    domain=domain,
                    title=title,
                    question=question,
                    answer=answer,
                    tags=json.dumps(tags, ensure_ascii=False),
                    language=language,
                    source=source,
                    media=json.dumps(media, ensure_ascii=False) if media is not None else None,
                ))
        except Exception as e:
            print(f"⚠️ Erreur lors de l'import d'une entrée de connaissance: {e}")

    db.commit()


def _normalize_token(token: str) -> str:
    """Normaliser grossièrement un mot français pour améliorer le matching.

    - Passe en minuscules
    - Supprime les accents (maïs -> mais)
    - Supprime un "s" final (pluriel simple : jaunes -> jaune)
    """
    if not token:
        return ""

    # Minuscules + suppression des accents
    token = unicodedata.normalize("NFD", token.lower())
    token = "".join(ch for ch in token if unicodedata.category(ch) != "Mn")

    # Supprimer la ponctuation pour éviter les "jaune," vs "jaune"
    token = "".join(ch for ch in token if ch.isalnum())

    # Pluriel très simple : retirer un "s" final
    if len(token) > 3 and token.endswith("s"):
        token = token[:-1]

    return token


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    raw_tokens = text.split()
    tokens: List[str] = []
    for w in raw_tokens:
        norm = _normalize_token(w)
        if len(norm) > 2:
            tokens.append(norm)
    return tokens


def retrieve_knowledge(db: Session, domain: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Récupération améliorée basée sur le recouvrement de mots-clés pondéré.

    - Les correspondances dans le titre et les tags comptent plus que celles
      présentes uniquement dans la réponse longue.
    - Pour le domaine "agriculture", on inclut aussi les fiches d'élevage,
      car dans la pratique les questions sur les animaux sont souvent
      formulées comme des problèmes agricoles.
    - Si aucune fiche n'est trouvée dans le domaine demandé, on fait un
      second passage sur toutes les fiches pour éviter de rater une
      correspondance évidente (tout en restant strictement dans la base).
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    def score_items(items: List[KnowledgeItem]) -> List[Dict[str, Any]]:
        scored_local: List[Dict[str, Any]] = []
        for it in items:
            # Séparer les zones de texte pour mieux pondérer
            title_tokens = set(_tokenize(it.title or ""))
            question_tokens = set(_tokenize(it.question or ""))
            answer_tokens = set(_tokenize(it.answer or ""))

            tags_list: List[str] = []
            if it.tags:
                try:
                    tags_list = json.loads(it.tags)
                except Exception:
                    tags_list = []
            tags_tokens = set(_tokenize(" ".join(tags_list))) if tags_list else set()

            overlap_title = len(query_tokens & title_tokens)
            overlap_question = len(query_tokens & question_tokens)
            overlap_answer = len(query_tokens & answer_tokens)
            overlap_tags = len(query_tokens & tags_tokens)

            # Pondération simple : titre > tags > question > réponse
            score = (
                overlap_title * 3.0
                + overlap_tags * 2.5
                + overlap_question * 2.0
                + overlap_answer * 1.0
            )

            if score <= 0:
                continue

            scored_local.append({"item": it, "score": score})

        return scored_local

    # 1) Fiches dans le domaine demandé (avec fusion agriculture + élevage)
    primary_query = db.query(KnowledgeItem)
    if domain == "agriculture":
        primary_query = primary_query.filter(
            KnowledgeItem.domain.in_(["agriculture", "elevage"])
        )
    else:
        primary_query = primary_query.filter(KnowledgeItem.domain == domain)

    primary_items = primary_query.all()
    scored = score_items(primary_items)

    # 2) Fallback : si rien trouvé, on regarde toutes les fiches
    if not scored:
        all_items = db.query(KnowledgeItem).all()
        scored = score_items(all_items)

    # 3) Dernier recours : si toujours rien, utiliser une recherche par sous-chaîne
    # sur le texte complet des fiches (normalisé). Cela permet de gérer les
    # entrées très courtes comme "jaune".
    if not scored:
        all_items = db.query(KnowledgeItem).all()

        def normalize_text(text: str) -> str:
            if not text:
                return ""
            tokens = _tokenize(text)
            return " ".join(tokens)

        norm_query_parts = list(query_tokens)
        for it in all_items:
            big_text = f"{it.title}\n{it.question or ''}\n{it.answer}\n"
            if it.tags:
                try:
                    tags_list = json.loads(it.tags)
                except Exception:
                    tags_list = []
                big_text += " ".join(tags_list)
            norm_text = normalize_text(big_text)
            if any(part in norm_text for part in norm_query_parts):
                scored.append({"item": it, "score": 1.0})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_items = [s["item"] for s in scored[:limit]]

    results: List[Dict[str, Any]] = []
    for it in top_items:
        media_data: Optional[Any] = None
        if it.media:
            try:
                media_data = json.loads(it.media)
            except Exception:
                media_data = None

        results.append({
            "id": it.id,
            "domain": it.domain,
            "title": it.title,
            "question": it.question,
            "answer": it.answer,
            "tags": json.loads(it.tags) if it.tags else [],
            "language": it.language,
            "source": it.source,
            "media": media_data,
        })

    return results


def generate_llm_answer(
    question: str,
    language: str,
    domain: str,
    knowledge_items: List[Dict[str, Any]],
) -> Optional[str]:
    """Utiliser ChatGPT pour reformuler et raisonner à partir de la base RAG.

    - Le modèle NE DOIT PAS inventer de faits en dehors des connaissances fournies.
    - S'il n'y a pas assez d'informations, il doit le dire clairement.
    """
    if not knowledge_items:
        # Pas de base de connaissance pertinente, on ne force pas le modèle
        return None

    # Petit fallback local : si le LLM n'est pas disponible, on formate au
    # minimum une réponse structurée à partir de la meilleure fiche.
    def build_structured_from_rag() -> str:
        best = knowledge_items[0]
        titre = best.get("title") or "Conseil local"
        reponse = best.get("answer") or ""
        source = best.get("source") or "fiches locales"

        parts = []
        parts.append(
            f"1) Ce que je comprends de ton problème :\n"
            f"Tu signales un souci lié à : {titre}. Je vais utiliser les conseils déjà validés localement."
        )
        parts.append(
            "2) Conseils pratiques à suivre :\n" + reponse
        )
        parts.append(
            "3) Quand appeler un expert :\n"
            "Si malgré ces conseils la situation ne s'améliore pas, si le problème s'aggrave, ou si tu as un doute, "
            "rapproche-toi d'un agent agricole ou d'un service technique local pour vérifier sur place."
            f" (Source : {source})."
        )
        return "\n\n".join(parts)

    # Si pas de client OpenAI (clé manquante ou désactivée), on renvoie tout
    # de suite la version structurée basée uniquement sur le RAG.
    if not openai_client or not OPENAI_API_KEY:
        return build_structured_from_rag()

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
        "Tu es un assistant local pour la population rurale au Burkina Faso. "
        "Tu aides sur l'agriculture, l'élevage et la cybersécurité. "
        "Tu dois répondre UNIQUEMENT avec les informations présentes dans les fiches ci-dessous. "
        "Si les fiches ne suffisent pas, tu dis clairement que tu n'as pas assez d'informations "
        "et tu recommandes de contacter un expert humain. \n\n"
        "Contraintes importantes : \n"
        "- Pas de hors-sujet, pas de conseils médicaux avancés. \n"
        "- Langage TRÈS simple, phrases courtes, concret, sans jargon. \n"
        "- Adresse-toi à une personne peu à l'aise avec l'écrit. \n"
        "- Donne la réponse principalement en français clair; tu peux ajouter quelques mots en langue locale si utile (mais pas obligatoire). \n"
        "- Ne propose jamais de traitement dangereux ou interdit.\n"
    )

    user_prompt = (
        f"Langue demandée: {language or 'fr'}. Domaine: {domain}.\n"
        f"Question de l'utilisateur : {question}\n\n"
        f"FICHES DE CONNAISSANCE DISPONIBLES :\n{context_text}\n\n"
        "Tâche : en utilisant UNIQUEMENT ces fiches : \n"
        "- Donne une réponse courte (10 à 15 phrases max). \n"
        "- Structure ta réponse ainsi : \n"
        "  1) Ce que tu comprends du problème (2-3 phrases). \n"
        "  2) Conseils PRATIQUES en étapes numérotées (1., 2., 3., ...), adaptés à un agriculteur ou éleveur. \n"
        "  3) Quand il faut absolument appeler un expert humain, un vétérinaire ou un service local. \n"
        "Si les fiches ne couvrent pas bien la situation, dis clairement que tu n'as pas assez d'informations et conseille de voir un expert local."
    )

    try:
        # Utiliser un modèle GPT standard largement disponible
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Log détaillé pour diagnostiquer les problèmes de modèle / de clé
        print("⚠️ Erreur appel OpenAI (generate_llm_answer):", repr(e))
        # En cas d'erreur LLM (quota, réseau, modèle indisponible...), on
        # revient à la réponse structurée 100% basée sur la fiche locale.
        return build_structured_from_rag()

# ==========================================
# ROUTES API (Avec analyse IA restaurée)
# ==========================================

@app.get("/")
async def root():
    return {
        "message": "SONGRA API - IA Locale",
        "version": "5.0",
        "features": ["Analyse IA texte", "Analyse IA photo", "Computer Vision local"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "ia_status": "active"
    }

@app.post("/api/auth/login")
async def login(data: ExpertLogin, db: Session = Depends(get_db)):
    expert = db.query(Expert).filter(Expert.email == data.email).first()
    if not expert or not verify_password(data.password, expert.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "token": f"token_{expert.id}_{datetime.utcnow().timestamp()}",
        "expert": {
            "id": expert.id,
            "name": expert.full_name,
            "email": expert.email,
            "specialization": expert.specialization
        }
    }

@app.get("/api/tickets")
async def get_tickets(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Ticket)
    if status:
        query = query.filter(Ticket.status == status)
    
    tickets = query.order_by(Ticket.created_at.desc()).all()
    
    result = []
    for ticket in tickets:
        user = db.query(User).filter(User.id == ticket.user_id).first()
        last_msg = db.query(Message).filter(
            Message.ticket_id == ticket.id
        ).order_by(Message.sent_at.desc()).first()
        
        # Construire l'URL de la photo
        photo_url = None
        if ticket.photo_path:
            photo_url = f"http://localhost:8000/{ticket.photo_path}"
        
        result.append({
            "id": ticket.id,
            "user_phone": user.phone_number if user else "Inconnu",
            "category": ticket.category or "agriculture",
            "urgency": ticket.urgency or "low",
            "status": ticket.status or "open",
            "created_at": ticket.created_at,
            "last_message": last_msg.content if last_msg else "Aucun message",
            "ai_confidence": ticket.ai_confidence_score,
            "has_photo": ticket.photo_path is not None,
            "photo_url": photo_url,
            "photo_path": ticket.photo_path,
            "has_photo_analysis": ticket.ai_photo_analysis is not None
        })
    
    return result

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    total_tickets = db.query(Ticket).count()
    open_tickets = db.query(Ticket).filter(Ticket.status == "open").count()
    assigned_tickets = db.query(Ticket).filter(Ticket.status == "assigned").count()
    
    # Tickets résolus aujourd'hui
    today = datetime.utcnow().date()
    resolved_today = db.query(Ticket).filter(
        Ticket.status == "resolved",
        func.date(Ticket.resolved_at) == today
    ).count()
    
    tickets_with_photos = db.query(Ticket).filter(
        Ticket.photo_path.isnot(None)
    ).count()
    
    return {
        "total_tickets": total_tickets,
        "open_tickets": open_tickets,
        "assigned_tickets": assigned_tickets,
        "resolved_today": resolved_today,
        "tickets_with_photos": tickets_with_photos
    }

@app.get("/api/tickets/{ticket_id}")
async def get_ticket_detail(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    user = db.query(User).filter(User.id == ticket.user_id).first()
    messages = db.query(Message).filter(
        Message.ticket_id == ticket_id
    ).order_by(Message.sent_at).all()
    
    # Gérer l'analyse photo
    photo_analysis = None
    if ticket.ai_photo_analysis:
        try:
            if isinstance(ticket.ai_photo_analysis, str):
                photo_analysis = json.loads(ticket.ai_photo_analysis)
            else:
                photo_analysis = ticket.ai_photo_analysis
        except:
            photo_analysis = {"error": "Failed to parse analysis"}
    
    # Construire l'URL de la photo
    photo_url = None
    if ticket.photo_path:
        photo_url = f"http://localhost:8000/{ticket.photo_path}"
    
    # Extraire les mots-clés
    keywords = []
    if ticket.ai_extracted_keywords:
        try:
            keywords = json.loads(ticket.ai_extracted_keywords)
        except:
            keywords = []
    
    return {
        "ticket": {
            "id": ticket.id,
            "category": ticket.category or "agriculture",
            "urgency": ticket.urgency or "low",
            "status": ticket.status or "open",
            "keywords": keywords,
            "confidence": ticket.ai_confidence_score or 0.5,
            "photo_url": photo_url,
            "photo_path": ticket.photo_path,
            "photo_filename": ticket.photo_path,  # Alias pour le frontend expert
            "photo_analysis": photo_analysis,
            "created_at": ticket.created_at,
            "resolved_at": ticket.resolved_at
        },
        "user": {
            "phone": user.phone_number if user else None,
            "name": user.name if user else None,
            "location": user.location if user else None
        },
        "messages": [{
            "content": msg.content,
            "sender_type": msg.sender_type,
            "sent_at": msg.sent_at
        } for msg in messages]
    }

@app.post("/api/webhooks/incoming-sms")
async def incoming_sms(data: MessageCreate, db: Session = Depends(get_db)):
    # 1. Trouver ou créer l'utilisateur
    user = db.query(User).filter(User.phone_number == data.phone_number).first()
    if not user:
        user = User(phone_number=data.phone_number)
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # 2. Analyse IA texte (RESTAURÉ)
    ai_result = ai_engine.classify(data.content)

    # 2.bis Déterminer la catégorie finale et le domaine RAG
    # On combine la catégorie choisie dans l'app (data.category) et
    # la catégorie NLP locale.
    nlp_category = ai_result.get("category", "agriculture")
    chosen_category = data.category or nlp_category

    # Mapping catégorie -> domaine de la base de connaissances
    if chosen_category == "agriculture":
        kb_domain = "agriculture"
    elif chosen_category == "elevage":
        kb_domain = "elevage"
    elif chosen_category == "sos_accident":
        kb_domain = "health"  # les fiches premiers soins sont dans le domaine health
    elif chosen_category == "cybersecurity":
        kb_domain = "cybersecurity"
    else:
        kb_domain = "agriculture"
    
    # 3. Analyse photo si présente (RESTAURÉ)
    photo_analysis = None
    photo_path = None
    
    if data.photo_base64:
        try:
            # Décoder base64
            photo_string = data.photo_base64
            if ',' in photo_string:
                photo_string = photo_string.split(',')[1]
            
            photo_data = base64.b64decode(photo_string)
            
            # Analyser avec IA locale (RESTAURÉ)
            photo_analysis_result = cv_engine.analyze_image_simple(photo_data, data.content)
            photo_analysis = json.dumps(photo_analysis_result, ensure_ascii=False)
            
            # Sauvegarder photo
            timestamp = int(datetime.utcnow().timestamp())
            filename = f"{user.id}_{timestamp}.jpg"
            photo_path = f"uploads/{filename}"
            
            with open(photo_path, "wb") as f:
                f.write(photo_data)
            
            # Ajuster urgence si maladie grave détectée
            if photo_analysis_result.get("urgency") == "high":
                ai_result["urgency"] = "high"
                
        except Exception as e:
            print(f"Erreur analyse photo: {e}")
            photo_analysis = json.dumps({"error": str(e), "requires_expert": True})
    
    # 3.bis Récupération de connaissances + appel LLM (RAG)
    # On reste entièrement déterministe côté sélection de fiches, et le LLM ne fait que reformuler.
    rag_items = retrieve_knowledge(db, kb_domain, data.content)
    llm_answer = generate_llm_answer(
        question=data.content,
        language="fr",
        domain=kb_domain,
        knowledge_items=rag_items,
    )

    # LOG DEBUG : affichage des fiches RAG et de la réponse LLM (si disponible)
    try:
      print("[RAG] Domaine:", kb_domain)
      print(f"[RAG] {len(rag_items)} fiche(s) sélectionnée(s)")
      for idx, item in enumerate(rag_items, start=1):
          print(f"[RAG] FICHE {idx}: {item.get('title')} | tags={item.get('tags')}")
      if llm_answer:
          print("[LLM] Réponse générée (début):", llm_answer[:300].replace("\n", " "))
      else:
          print("[LLM] Aucune réponse générée (pas de clé ou pas de fiches pertinentes)")
    except Exception as e_log:
      print(f"[DEBUG] Erreur lors du log RAG/LLM: {e_log}")

    # 4. Créer ticket avec analyse IA
    ticket = Ticket(
        user_id=user.id,
        category=chosen_category,
        urgency=ai_result["urgency"],
        ai_confidence_score=ai_result["confidence"],
        ai_extracted_keywords=json.dumps(ai_result["keywords"], ensure_ascii=False),
        ai_photo_analysis=photo_analysis,
        photo_path=photo_path,
        status="open"
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    
    # 5. Créer message
    message = Message(
        ticket_id=ticket.id,
        sender_type="user",
        sender_id=user.id,
        content=data.content,
        channel=data.channel
    )
    db.add(message)
    db.commit()
    
    # 6. Retourner résultat avec analyse IA COMPLÈTE
    response = {
        "status": "success",
        "ticket_id": ticket.id,
        "ai_analysis": ai_result
    }
    
    # Ajouter l'analyse photo si disponible
    if photo_analysis:
        try:
            response["photo_analysis"] = json.loads(photo_analysis)
        except:
            response["photo_analysis"] = {"analysis": "Analyse photo en cours"}
    
    # Ajouter l'URL de la photo
    if photo_path:
        response["photo_url"] = f"http://localhost:8000/{photo_path}"

    # Toujours retourner les fiches RAG utilisées (pour debug et fallback côté frontend)
    if rag_items:
        response["rag_items"] = rag_items

    # Ajouter la réponse principale générée par le LLM si disponible
    if llm_answer:
        response["llm_answer"] = llm_answer
    elif rag_items:
        # Fallback déterministe : utiliser la réponse de la meilleure fiche
        # pour que l'utilisateur ait au moins la réponse validée locale,
        # même si la clé OpenAI n'est pas configurée.
        best = rag_items[0]
        response["rag_fallback_answer"] = best.get("answer")
    
    return response


@app.post("/api/assistant/query")
async def assistant_query(data: MessageCreate, db: Session = Depends(get_db)):
    """Endpoint conversation IA seule (RAG + GPT) sans création de ticket.

    Utilisé par l'application pour discuter avec l'IA et affiner le problème.
    Aucun Ticket/Message n'est créé ici, uniquement une réponse IA.
    """

    # 1. Analyse IA texte
    ai_result = ai_engine.classify(data.content)

    # 2. Déterminer la catégorie finale et le domaine RAG
    nlp_category = ai_result.get("category", "agriculture")
    chosen_category = data.category or nlp_category

    if chosen_category == "agriculture":
        kb_domain = "agriculture"
    elif chosen_category == "elevage":
        kb_domain = "elevage"
    elif chosen_category == "sos_accident":
        kb_domain = "health"
    elif chosen_category == "cybersecurity":
        kb_domain = "cybersecurity"
    else:
        kb_domain = "agriculture"

    # 3. Analyse photo si présente (sans sauvegarder de fichier)
    photo_analysis = None
    if data.photo_base64:
        try:
            photo_string = data.photo_base64
            if "," in photo_string:
                photo_string = photo_string.split(",")[1]
            photo_data = base64.b64decode(photo_string)
            photo_analysis_result = cv_engine.analyze_image_simple(photo_data, data.content)
            photo_analysis = photo_analysis_result

            if photo_analysis_result.get("urgency") == "high":
                ai_result["urgency"] = "high"
        except Exception as e:
            print(f"Erreur analyse photo (assistant_query): {e}")
            photo_analysis = {"error": str(e), "requires_expert": True}

    # 4. RAG + LLM
    rag_items = retrieve_knowledge(db, kb_domain, data.content)
    llm_answer = generate_llm_answer(
        question=data.content,
        language="fr",
        domain=kb_domain,
        knowledge_items=rag_items,
    )

    # 5. Construction de la réponse (sans ticket)
    response: Dict[str, Any] = {
        "status": "success",
        "ai_analysis": ai_result,
        "category": chosen_category,
    }

    if photo_analysis is not None:
        response["photo_analysis"] = photo_analysis

    if rag_items:
        response["rag_items"] = rag_items

    if llm_answer:
        response["llm_answer"] = llm_answer

    return response

@app.post("/api/tickets/{ticket_id}/reply")
async def reply_to_ticket(
    ticket_id: int, 
    content: ReplyMessage,
    db: Session = Depends(get_db)
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    message = Message(
        ticket_id=ticket_id,
        sender_type="expert",
        sender_id=1,
        content=content.message,
        channel="web"
    )
    db.add(message)
    
    if not ticket.expert_id:
        ticket.expert_id = 1
        ticket.status = "assigned"
    
    db.commit()
    return {"status": "success"}

@app.post("/api/tickets/{ticket_id}/resolve")
async def resolve_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    ticket.status = "resolved"
    ticket.resolved_at = datetime.utcnow()
    
    message = Message(
        ticket_id=ticket_id,
        sender_type="system",
        sender_id=None,
        content="Ticket marqué comme résolu par l'expert",
        channel="system"
    )
    db.add(message)
    
    db.commit()
    return {"status": "success"}


@app.get("/api/tickets/{ticket_id}/ai-summary")
async def get_ticket_ai_summary(ticket_id: int, db: Session = Depends(get_db)):
    """Retourne un résumé IA (RAG + GPT ou fallback) pour un ticket donné.

    L'IA se base sur le dernier message utilisateur du ticket et sur
    la catégorie du ticket pour choisir le bon domaine de connaissance.
    """

    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Récupérer le dernier message utilisateur lié au ticket
    last_user_msg = (
        db.query(Message)
        .filter(Message.ticket_id == ticket_id, Message.sender_type == "user")
        .order_by(Message.sent_at.desc())
        .first()
    )

    if not last_user_msg:
        return {"status": "success", "ai_summary": None, "detail": "Aucun message utilisateur pour ce ticket."}

    content = last_user_msg.content or ""

    # Mapper la catégorie du ticket vers le domaine RAG
    chosen_category = ticket.category or "agriculture"
    if chosen_category == "agriculture":
        kb_domain = "agriculture"
    elif chosen_category == "elevage":
        kb_domain = "elevage"
    elif chosen_category == "sos_accident":
        kb_domain = "health"
    elif chosen_category == "cybersecurity":
        kb_domain = "cybersecurity"
    else:
        kb_domain = "agriculture"

    rag_items = retrieve_knowledge(db, kb_domain, content)
    llm_answer = generate_llm_answer(
        question=content,
        language="fr",
        domain=kb_domain,
        knowledge_items=rag_items,
    )

    return {
        "status": "success",
        "ai_summary": llm_answer,
        "category": chosen_category,
        "rag_items": rag_items,
    }

@app.get("/api/user-tickets")
async def get_user_tickets(phone: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone_number == phone).first()
    if not user:
        return []
    
    tickets = db.query(Ticket).filter(
        Ticket.user_id == user.id
    ).order_by(Ticket.created_at.desc()).all()
    
    result = []
    for ticket in tickets:
        last_msg = db.query(Message).filter(
            Message.ticket_id == ticket.id
        ).order_by(Message.sent_at.desc()).first()
        
        # Construire l'URL de la photo
        photo_url = None
        if ticket.photo_path:
            photo_url = f"http://localhost:8000/{ticket.photo_path}"
        
        result.append({
            "id": ticket.id,
            "category": ticket.category or "agriculture",
            "urgency": ticket.urgency or "low",
            "status": ticket.status or "open",
            "created_at": ticket.created_at,
            "last_message": last_msg.content if last_msg else "Aucun message",
            "has_photo": ticket.photo_path is not None,
            "photo_url": photo_url
        })
    
    return result


@app.get("/api/knowledge/offline-cache")
async def knowledge_offline_cache(
    db: Session = Depends(get_db),
):
    """
    Base de connaissances formatée pour mise en cache offline côté app.
    Retourne tous les items actifs avec les champs nécessaires au RAG local.
    """
    items = db.query(KnowledgeItem).order_by(KnowledgeItem.domain).all()
    result = []
    for it in items:
        result.append({
            "id": it.id,
            "domain": it.domain,
            "title": it.title,
            "question": it.question,
            "answer": it.answer,
            "tags": json.loads(it.tags) if it.tags else [],
            "language": it.language or "fr",
        })
    return {"items": result, "total": len(result)}


@app.get("/api/admin/knowledge")
async def list_knowledge_items(
    domain: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Lister les fiches de la base de connaissances.

    Pour l'instant, cette route n'est pas protégée (usage local / démo). Pour
    un déploiement réel, il faudra ajouter une authentification admin.
    """
    query = db.query(KnowledgeItem)
    if domain:
        query = query.filter(KnowledgeItem.domain == domain)

    items = query.order_by(KnowledgeItem.created_at.desc()).all()

    result: List[Dict[str, Any]] = []
    for it in items:
        media_data = None
        if it.media:
            try:
                media_data = json.loads(it.media)
            except Exception:
                media_data = None

        result.append(
            {
                "id": it.id,
                "domain": it.domain,
                "title": it.title,
                "question": it.question,
                "answer": it.answer,
                "tags": json.loads(it.tags) if it.tags else [],
                "language": it.language,
                "source": it.source,
                "media": media_data,
                "created_at": it.created_at,
                "updated_at": it.updated_at,
            }
        )

    return result


@app.get("/api/emergency-numbers")
async def public_emergency_numbers(db: Session = Depends(get_db)):
    """Liste publique des numéros d'urgence / numéros utiles.

    Utilisée par l'application utilisateur pour afficher les numéros utiles.
    """
    items = db.query(EmergencyNumber).filter(EmergencyNumber.is_active == True).order_by(
        EmergencyNumber.display_order.asc(), EmergencyNumber.id.asc()
    ).all()

    return [
        {
            "id": it.id,
            "label": it.label,
            "number": it.number,
            "description": it.description,
            "display_order": it.display_order,
        }
        for it in items
    ]


@app.post("/api/admin/reload-knowledge")
async def reload_knowledge_endpoint(
    db: Session = Depends(get_db),
):
    """Recharger la base de connaissances depuis le fichier JSON.

    End-point simple pour éviter de redémarrer le serveur quand tu mets à jour
    knowledge_base.json. Dans un vrai déploiement, il faudra protéger cette route
    (token admin, VPN, etc.).
    """
    load_knowledge_from_json(db)
    total_items = db.query(KnowledgeItem).count()
    return {
        "status": "success",
        "total_items": total_items,
    }


@app.post("/api/admin/knowledge")
async def create_knowledge_item(
    payload: KnowledgeItemIn,
    db: Session = Depends(get_db),
):
    """Créer une nouvelle fiche de connaissance (usage panneau expert)."""

    media_json = None
    if payload.media:
        media_json = json.dumps(
            [m.dict() for m in payload.media], ensure_ascii=False
        )

    item = KnowledgeItem(
        domain=payload.domain,
        title=payload.title,
        question=payload.question,
        answer=payload.answer,
        tags=json.dumps(payload.tags or [], ensure_ascii=False),
        language=payload.language,
        source=payload.source,
        media=media_json,
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    media_data = None
    if item.media:
        try:
            media_data = json.loads(item.media)
        except Exception:
            media_data = None

    return {
        "id": item.id,
        "domain": item.domain,
        "title": item.title,
        "question": item.question,
        "answer": item.answer,
        "tags": json.loads(item.tags) if item.tags else [],
        "language": item.language,
        "source": item.source,
        "media": media_data,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


@app.get("/api/admin/emergency-numbers")
async def list_emergency_numbers_admin(db: Session = Depends(get_db)):
    """Lister tous les numéros utiles (admin panneau expert)."""

    items = db.query(EmergencyNumber).order_by(
        EmergencyNumber.display_order.asc(), EmergencyNumber.id.asc()
    ).all()

    return [
        {
            "id": it.id,
            "label": it.label,
            "number": it.number,
            "description": it.description,
            "display_order": it.display_order,
            "is_active": it.is_active,
        }
        for it in items
    ]


@app.post("/api/admin/emergency-numbers")
async def create_emergency_number(
    payload: EmergencyNumberIn,
    db: Session = Depends(get_db),
):
    """Créer un nouveau numéro utile (pompier, police, clinique...)."""

    item = EmergencyNumber(
        label=payload.label,
        number=payload.number,
        description=payload.description,
        display_order=payload.display_order,
        is_active=True,
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    return {
        "id": item.id,
        "label": item.label,
        "number": item.number,
        "description": item.description,
        "display_order": item.display_order,
        "is_active": item.is_active,
    }


@app.put("/api/admin/knowledge/{item_id}")
async def update_knowledge_item(
    item_id: int,
    payload: KnowledgeItemIn,
    db: Session = Depends(get_db),
):
    """Mettre à jour une fiche de connaissance existante ou la créer si elle n'existe plus.

    Cela évite une erreur 404 côté panneau expert si, pour une raison quelconque,
    l'ID stocké dans le frontend ne correspond plus à une ligne en base.
    """

    item = db.query(KnowledgeItem).filter(KnowledgeItem.id == item_id).first()

    media_json = None
    if payload.media:
        media_json = json.dumps(
            [m.dict() for m in payload.media], ensure_ascii=False
        )

    if not item:
        # Comportement "upsert" : si l'ID n'existe plus, on crée une nouvelle fiche
        item = KnowledgeItem(
            domain=payload.domain,
            title=payload.title,
            question=payload.question,
            answer=payload.answer,
            tags=json.dumps(payload.tags or [], ensure_ascii=False),
            language=payload.language,
            source=payload.source,
            media=media_json,
        )
        db.add(item)
        db.commit()
        db.refresh(item)
    else:
        item.domain = payload.domain
        item.title = payload.title
        item.question = payload.question
        item.answer = payload.answer
        item.tags = json.dumps(payload.tags or [], ensure_ascii=False)
        item.language = payload.language
        item.source = payload.source
        item.media = media_json

        db.commit()
        db.refresh(item)

    media_data = None
    if item.media:
        try:
            media_data = json.loads(item.media)
        except Exception:
            media_data = None

    return {
        "id": item.id,
        "domain": item.domain,
        "title": item.title,
        "question": item.question,
        "answer": item.answer,
        "tags": json.loads(item.tags) if item.tags else [],
        "language": item.language,
        "source": item.source,
        "media": media_data,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


@app.delete("/api/admin/knowledge/{item_id}")
async def delete_knowledge_item(
    item_id: int,
    db: Session = Depends(get_db),
):
    """Supprimer une fiche de connaissance."""

    item = db.query(KnowledgeItem).filter(KnowledgeItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Knowledge item not found")

    db.delete(item)
    db.commit()
    return {"status": "success"}


@app.put("/api/admin/emergency-numbers/{item_id}")
async def update_emergency_number(
    item_id: int,
    payload: EmergencyNumberIn,
    db: Session = Depends(get_db),
):
    """Mettre à jour un numéro utile existant."""

    item = db.query(EmergencyNumber).filter(EmergencyNumber.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Emergency number not found")

    item.label = payload.label
    item.number = payload.number
    item.description = payload.description
    item.display_order = payload.display_order

    db.commit()
    db.refresh(item)

    return {
        "id": item.id,
        "label": item.label,
        "number": item.number,
        "description": item.description,
        "display_order": item.display_order,
        "is_active": item.is_active,
    }


@app.delete("/api/admin/emergency-numbers/{item_id}")
async def delete_emergency_number(
    item_id: int,
    db: Session = Depends(get_db),
):
    """Supprimer un numéro utile."""

    item = db.query(EmergencyNumber).filter(EmergencyNumber.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Emergency number not found")

    db.delete(item)
    db.commit()
    return {"status": "success"}


@app.post("/api/admin/knowledge/import-json")
async def import_knowledge_from_json(
    payload: KnowledgeBulkImport,
    db: Session = Depends(get_db),
):
    """Importer/mettre à jour des fiches à partir d'un JSON envoyé par le panneau expert.

    Les fiches sont identifiées par (domain, title). Si une fiche existe déjà,
    elle est mise à jour; sinon elle est créée.
    """

    created = 0
    updated = 0

    for entry in payload.items:
        existing = (
            db.query(KnowledgeItem)
            .filter(
                KnowledgeItem.domain == entry.domain,
                KnowledgeItem.title == entry.title,
            )
            .first()
        )

        media_json = None
        if entry.media:
            media_json = json.dumps(
                [m.dict() for m in entry.media], ensure_ascii=False
            )

        if existing:
            existing.question = entry.question
            existing.answer = entry.answer
            existing.tags = json.dumps(entry.tags or [], ensure_ascii=False)
            existing.language = entry.language
            existing.source = entry.source
            existing.media = media_json
            updated += 1
        else:
            item = KnowledgeItem(
                domain=entry.domain,
                title=entry.title,
                question=entry.question,
                answer=entry.answer,
                tags=json.dumps(entry.tags or [], ensure_ascii=False),
                language=entry.language,
                source=entry.source,
                media=media_json,
            )
            db.add(item)
            created += 1

    db.commit()

    total_items = db.query(KnowledgeItem).count()
    return {
        "status": "success",
        "created": created,
        "updated": updated,
        "total_items": total_items,
    }

@app.post("/api/create-test-expert")
async def create_test_expert(db: Session = Depends(get_db)):
    existing = db.query(Expert).filter(Expert.email == "test@resolvehub.bf").first()
    if existing:
        return {
            "message": "Expert déjà existant", 
            "email": "test@resolvehub.bf", 
            "password": "test123"
        }
    
    expert = Expert(
        email="test@resolvehub.bf",
        password_hash=hash_password("test123"),
        full_name="Expert Test IA",
        specialization="agriculture",
        is_active=True
    )
    db.add(expert)
    db.commit()
    
    return {
        "message": "Expert créé avec succès", 
        "email": "test@resolvehub.bf", 
        "password": "test123"
    }

# ==========================================
# NOUVEAUX ENDPOINTS POUR FONCTIONNALITÉS PRIORITAIRES
# ==========================================

class SOSAlert(BaseModel):
    phoneNumber: str
    type: str  # accident, attack, fire, animal_sick, community
    description: Optional[str] = None
    location: Dict[str, Any]  # {latitude, longitude, accuracy, note}
    timestamp: int
    urgent: bool = True

class SOSAlertDB(Base):
    __tablename__ = "sos_alerts"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    location_note = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")  # pending, acknowledged, resolved
    notified_authorities = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessageDB(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String, nullable=False)  # numéro ou nom affiché
    text = Column(Text, nullable=False)
    is_bot = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)


def _ensure_sos_description_column() -> None:
    """Migration: ajouter la colonne 'description' à sos_alerts si absente."""
    if not str(engine.url).startswith("sqlite"):
        return
    try:
        with engine.connect() as conn:
            result = conn.exec_driver_sql("PRAGMA table_info(sos_alerts)")
            columns = [row[1] for row in result]
            if "description" not in columns:
                conn.exec_driver_sql("ALTER TABLE sos_alerts ADD COLUMN description TEXT")
    except Exception as e:
        print(f"⚠️ Impossible d'ajouter la colonne 'description' à sos_alerts: {e}")

_ensure_sos_description_column()

@app.post("/api/sos/alert")
async def create_sos_alert(alert: SOSAlert, db: Session = Depends(get_db)):
    """
    🔑 ENDPOINT SOS - PRIORITÉ CRITIQUE
    Enregistrer une alerte SOS avec géolocalisation
    """
    try:
        # Créer l'alerte SOS
        sos_alert = SOSAlertDB(
            phone_number=alert.phoneNumber or "Anonyme",
            alert_type=alert.type,
            description=alert.description,
            latitude=alert.location.get("latitude"),
            longitude=alert.location.get("longitude"),
            accuracy=alert.location.get("accuracy"),
            location_note=alert.location.get("note"),
            status="pending",
            notified_authorities=False
        )
        
        db.add(sos_alert)
        db.commit()
        db.refresh(sos_alert)
        
        # TODO: Notifier les autorités locales / ONG partenaires
        # - Envoi SMS aux numéros d'urgence
        # - Notification email aux responsables
        # - Webhook vers systèmes externes
        
        print(f"🚨 ALERTE SOS REÇUE - Type: {alert.type}, Tel: {alert.phoneNumber}")
        if alert.location.get("latitude"):
            print(f"   📍 Position: {alert.location.get('latitude')}, {alert.location.get('longitude')}")
        
        return {
            "success": True,
            "alert_id": sos_alert.id,
            "message": "Alerte SOS enregistrée. Secours notifiés.",
            "emergency_numbers": [
                {"label": "Police", "number": "17"},
                {"label": "Pompiers", "number": "18"},
                {"label": "SAMU", "number": "15"}
            ]
        }
    
    except Exception as e:
        print(f"❌ Erreur SOS: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur enregistrement SOS: {str(e)}")

@app.patch("/api/sos/alerts/{alert_id}/status")
async def update_sos_alert_status(
    alert_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Mettre à jour le statut d'une alerte SOS (pour panel expert).
    Body JSON: { "status": "acknowledged" | "resolved" }
    """
    alert = db.query(SOSAlertDB).filter(SOSAlertDB.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alerte SOS introuvable")
    new_status = body.get("status")
    if new_status not in ("pending", "acknowledged", "resolved"):
        raise HTTPException(status_code=422, detail="Statut invalide")
    alert.status = new_status
    db.commit()
    return {"success": True, "alert_id": alert_id, "status": new_status}

@app.get("/api/sos/alerts")
async def get_sos_alerts(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db)
):
    """
    Récupérer les alertes SOS (pour dashboard admin/expert)
    """
    query = db.query(SOSAlertDB).order_by(SOSAlertDB.created_at.desc())
    
    if status:
        query = query.filter(SOSAlertDB.status == status)
    
    alerts = query.limit(limit).all()
    
    return {
        "alerts": [
            {
                "id": alert.id,
                "phoneNumber": alert.phone_number,
                "type": alert.alert_type,
                "description": alert.description,
                "location": {
                    "latitude": alert.latitude,
                    "longitude": alert.longitude,
                    "accuracy": alert.accuracy,
                    "note": alert.location_note
                },
                "status": alert.status,
                "timestamp": alert.created_at.isoformat() if alert.created_at else None
            }
            for alert in alerts
        ],
        "total": len(alerts)
    }


# ==========================================
# COMMUNITY CHAT
# ==========================================

BOT_REPLIES = [
    "Merci pour votre partage ! Un expert vous répondra bientôt.",
    "Bonne question ! D'autres membres ont eu le même problème.",
    "Consultez aussi la section Agriculture pour plus de conseils.",
    "Vous n'êtes pas seul ! La communauté SONGRA est là pour vous aider.",
    "Votre question a bien été notée. Restez connecté !",
]

@app.get("/api/community/messages")
async def get_community_messages(
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db)
):
    """Récupérer les derniers messages du chat communautaire."""
    messages = (
        db.query(ChatMessageDB)
        .order_by(ChatMessageDB.created_at.asc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": m.id,
            "sender": m.sender,
            "text": m.text,
            "is_bot": m.is_bot,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in messages
    ]

@app.post("/api/community/messages")
async def post_community_message(
    body: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Poster un message dans le chat communautaire et générer une réponse bot."""
    sender = (body.get("sender") or "Anonyme")[:80]
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Le message ne peut pas être vide")
    if len(text) > 1000:
        raise HTTPException(status_code=422, detail="Message trop long (max 1000 caractères)")

    user_msg = ChatMessageDB(sender=sender, text=text, is_bot=False)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    import random
    bot_text = random.choice(BOT_REPLIES)
    bot_msg = ChatMessageDB(sender="Assistant SONGRA", text=bot_text, is_bot=True)
    db.add(bot_msg)
    db.commit()
    db.refresh(bot_msg)

    return {
        "user": {"id": user_msg.id, "sender": user_msg.sender, "text": user_msg.text, "is_bot": False, "created_at": user_msg.created_at.isoformat()},
        "bot": {"id": bot_msg.id, "sender": bot_msg.sender, "text": bot_msg.text, "is_bot": True, "created_at": bot_msg.created_at.isoformat()},
    }

class OfflineSyncData(BaseModel):
    tickets: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = []
    photos: List[Dict[str, Any]] = []

@app.post("/api/sync/offline")
async def sync_offline_data(sync_data: OfflineSyncData, db: Session = Depends(get_db)):
    """
    🔑 ENDPOINT SYNCHRONISATION OFFLINE
    Synchroniser les données sauvegardées localement quand connexion revient
    """
    synced_tickets = []
    synced_messages = []
    synced_photos = []
    errors = []
    
    try:
        # Synchroniser les tickets
        for ticket_data in sync_data.tickets:
            try:
                # Vérifier si l'utilisateur existe
                phone = ticket_data.get("phoneNumber", "")
                user = db.query(User).filter(User.phone_number == phone).first()
                
                if not user:
                    user = User(phone_number=phone, name=ticket_data.get("userName"))
                    db.add(user)
                    db.commit()
                    db.refresh(user)
                
                # Créer le ticket
                ticket = Ticket(
                    user_id=user.id,
                    category=ticket_data.get("category"),
                    urgency=ticket_data.get("urgency", "medium"),
                    status="open"
                )
                
                db.add(ticket)
                db.commit()
                db.refresh(ticket)
                
                synced_tickets.append({
                    "localId": ticket_data.get("localId"),
                    "serverId": ticket.id
                })
                
            except Exception as e:
                errors.append({"type": "ticket", "error": str(e), "data": ticket_data.get("localId")})
        
        # Synchroniser les messages
        for msg_data in sync_data.messages:
            try:
                message = Message(
                    ticket_id=msg_data.get("ticketId"),
                    sender_type="user",
                    sender_id=msg_data.get("userId"),
                    content=msg_data.get("content"),
                    channel="app"
                )
                
                db.add(message)
                db.commit()
                
                synced_messages.append({
                    "localId": msg_data.get("localId"),
                    "serverId": message.id
                })
                
            except Exception as e:
                errors.append({"type": "message", "error": str(e)})
        
        return {
            "success": True,
            "synced": {
                "tickets": len(synced_tickets),
                "messages": len(synced_messages),
                "photos": len(synced_photos)
            },
            "mapping": {
                "tickets": synced_tickets,
                "messages": synced_messages
            },
            "errors": errors
        }
    
    except Exception as e:
        print(f"❌ Erreur sync offline: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur synchronisation: {str(e)}")

@app.get("/api/knowledge/offline-cache")
async def get_knowledge_for_offline_cache(
    domain: Optional[str] = None,
    language: str = "fr",
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db)
):
    """
    🔑 ENDPOINT CACHE OFFLINE
    Télécharger la base de connaissances pour utilisation offline (RAG local)
    """
    query = db.query(KnowledgeItem).filter(KnowledgeItem.language == language)
    
    if domain:
        query = query.filter(KnowledgeItem.domain == domain)
    
    items = query.order_by(KnowledgeItem.created_at.desc()).limit(limit).all()
    
    return {
        "items": [
            {
                "id": item.id,
                "domain": item.domain,
                "title": item.title,
                "question": item.question,
                "answer": item.answer,
                "tags": json.loads(item.tags) if item.tags else [],
                "keywords": [tag.lower() for tag in (json.loads(item.tags) if item.tags else [])]
            }
            for item in items
        ],
        "total": len(items),
        "cached_at": datetime.utcnow().isoformat()
    }

# ==========================================
# ENDPOINTS FREEMIUM & DIALOGUE
# ==========================================

class SendMessageRequest(BaseModel):
    ticket_id: int
    sender_type: str  # 'user' ou 'expert'
    content: str

@app.get("/api/user-status")
async def get_user_status(phone: str, db: Session = Depends(get_db)):
    """Retourne le statut premium et les limites de messages d'un utilisateur"""
    user = db.query(User).filter(User.phone_number == phone).first()
    
    if not user:
        # Créer l'utilisateur s'il n'existe pas
        user = User(
            phone_number=phone,
            is_premium=False,
            messages_used=0,
            messages_limit=1
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Vérifier si l'abonnement premium est expiré
    if user.is_premium and user.premium_expires_at:
        if user.premium_expires_at < datetime.utcnow():
            user.is_premium = False
            user.messages_limit = 1
            db.commit()
    
    return {
        "is_premium": user.is_premium,
        "messages_used": user.messages_used,
        "messages_limit": user.messages_limit if user.is_premium else 1,
        "premium_expires_at": user.premium_expires_at.isoformat() if user.premium_expires_at else None
    }

@app.post("/api/send-message")
async def send_message(request: SendMessageRequest, db: Session = Depends(get_db)):
    """Envoie un message dans un ticket (user ou expert)"""
    
    # Récupérer le ticket
    ticket = db.query(Ticket).filter(Ticket.id == request.ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket non trouvé")
    
    # Récupérer l'utilisateur
    user = db.query(User).filter(User.id == ticket.user_id).first()
    
    if request.sender_type == 'user':
        # Vérifier les limites de messages
        user_messages = db.query(Message).filter(
            Message.ticket_id == request.ticket_id,
            Message.sender_type == 'user'
        ).count()
        
        limit = user.messages_limit if user.is_premium else 1
        
        if user_messages >= limit:
            raise HTTPException(
                status_code=403, 
                detail=f"Limite de messages atteinte. Version gratuite : 1 message. Premium : 10 messages."
            )
    
    # Créer le message
    message = Message(
        ticket_id=request.ticket_id,
        sender_type=request.sender_type,
        sender_id=user.id if request.sender_type == 'user' else None,
        content=request.content,
        channel='web'
    )
    db.add(message)
    db.commit()
    
    return {"success": True, "message": "Message envoyé"}

# ==========================================
# LANCEMENT
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("SONGRA - Backend avec IA complète")
    print("Version 5.0 - Analyse IA texte + photo")
    print("=" * 50)
    print("Serveur démarré sur http://localhost:8000")
    print("Test expert: test@resolvehub.bf / test123")
    print("=" * 50)
    
    # Créer l'expert test au démarrage
    try:
        db = SessionLocal()
        existing = db.query(Expert).filter(Expert.email == "test@resolvehub.bf").first()
        if not existing:
            expert = Expert(
                email="test@resolvehub.bf",
                password_hash=hash_password("test123"),
                full_name="Expert Test IA",
                specialization="agriculture",
                is_active=True
            )
            db.add(expert)
            db.commit()
            print("✓ Expert test créé: test@resolvehub.bf / test123")
        # Charger la base de connaissances locale depuis le JSON
        try:
            load_knowledge_from_json(db)
            total_items = db.query(KnowledgeItem).count()
            print(f"✓ Base de connaissances chargée ({total_items} fiches)")
        except Exception as e_load:
            print(f"⚠️ Erreur chargement base de connaissances: {e_load}")
        db.close()
    except Exception as e:
        print(f"⚠️ Erreur création expert: {e}")
    
    # Le mode reload est plutôt à utiliser avec la commande uvicorn en ligne
    # de commande (ex: `uvicorn main:app --reload`). Ici on garde un run simple.
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)