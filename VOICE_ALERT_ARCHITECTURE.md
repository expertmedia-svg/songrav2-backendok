# SONGRA Voice Alert — architecture de développement

Cette phase n'intègre aucune API Orange Voice réelle. `MockProvider` ne fait
aucun accès réseau et marque toutes ses réponses avec `simulation: true`.

Flux : `IncidentReport → AgriculturalIncident → validation expert → résolution
des producteurs → VoiceCampaign → TelecomJobDB → MockProvider → résultats`.

Les SOS humains restent dans `SOSAlertDB`. Les parcelles mobiles existantes
sont synchronisées vers `farm_plots`; elles ne sont pas remplacées. La file est
persistée dans `telecom_jobs`, traitée par lots bornés et possède les états
`QUEUED`, `PROCESSING`, `COMPLETED` et `DLQ`.

Le worker indépendant se lance avec `python voice_alert_worker.py`. Pour une
exécution unique de démonstration: `MOCK_WORKER_ONCE=true`. La tâche FastAPI
ne traite qu'un premier lot borné; le worker reste le mécanisme durable prévu.

PostgreSQL/PostGIS est la cible de production. SQLite reste accepté en
développement avec calcul Haversine. Une migration ultérieure utilisera des
colonnes GeoAlchemy/PostGIS et `ST_DWithin` lorsque l'extension sera disponible.

Les langues réutilisent les codes SONGRA existants: `fr`, `moore`, `dioula`,
`fulfulde`. Aucun second moteur de traduction n'est introduit.
