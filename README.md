# Backend Songra

API FastAPI pour les applications Songra. La branche `main` contient la version récente de l'assistant rural, de la base de connaissances, de l'Academy hors ligne et des alertes vocales.

## Installation native, sans Docker

Utiliser Python 3.11 pour les versions de dépendances épinglées dans ce dépôt.

```bash
git clone https://github.com/expertmedia-svg/songrav2-backendok.git
cd songrav2-backendok
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
# Renseigner les secrets et les origines autorisées dans .env avant de lancer.
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 3000
```

Pour un serveur public, utiliser un service systemd pour Uvicorn et un reverse proxy Nginx avec HTTPS. Le service doit avoir ce dossier comme `WorkingDirectory` et charger son fichier `.env`. Ne pas exposer directement Uvicorn sur Internet.

## Configuration IA

Groq est le fournisseur par défaut :

```dotenv
AI_PROVIDER=groq
GROQ_API_KEY=remplacer-sur-le-serveur
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_VISION_MODEL=qwen/qwen3.8-27b
```

Vérifier que les modèles configurés sont disponibles sur le compte Groq. En mode Groq, les clés Gemini et OpenAI ne sont pas chargées par les moteurs principaux et les requêtes ne basculent pas vers ces fournisseurs. Le paquet Python `openai` sert de client compatible pour l'URL **Groq** `https://api.groq.com/openai/v1` ; il ne signifie pas que le service OpenAI est utilisé.

Le texte, les images en entrée, les traductions, l'analyse entrepreneuriale et l'inférence v3 passent par Groq. La transcription utilise `whisper-large-v3-turbo` chez Groq. La génération de nouvelles images et vidéos n'est pas disponible dans cette configuration et retourne une indisponibilité explicite.

Configurer aussi un `JWT_SECRET` aléatoire, `ALLOWED_ORIGINS` et les services nécessaires à la présentation. Les clés de paiement ou SMS ne sont nécessaires que pour ces fonctionnalités. Ne jamais committer `.env`, les bases SQLite ou les fichiers utilisateurs.

## Données et vérification

SQLite est utilisé par défaut. Le code ne contient pas les comptes ni les fichiers du serveur officiel : leur restauration exige une sauvegarde. Préserver le fichier de base et `uploads/` pendant les mises à jour ; effectuer une sauvegarde cohérente SQLite avant une migration.

- `/health` : disponibilité de l'API.
- `/api/v2/health` : fournisseur et modèle configurés, sans clé secrète.
- `/api/dashboard` : tableau de bord mobile authentifié.

Tests sans appels IA réels :

```bash
AI_PROVIDER=groq .venv/bin/python -m pytest tests -q
```

Les tests utilisent des doubles pour les fournisseurs ; une vérification réelle avec le compte Groq reste nécessaire lors d'un déploiement.
