# SONGRA — pertinence des fiches et compréhension de la question

Correction locale du 7 octobre 2026. Aucun contenu de fiche modifié, aucune fiche
supprimée, aucun changement du design. La VM n'a pas été modifiée par ce travail.

## Cause reproduite, et pas seulement supposée

Le code du commit `fc85d24` a été rejoué sans fournisseur IA et sans écriture sur
`backend/resolvehub.db` : 42 fiches au total, dont 10 agricoles, aucune fiche
Studio ni aucun cours dans cette base locale. Les résultats sont conservés dans
`tests/relevance_legacy_baseline.json`. Script reproductible :
`python tests/replay_legacy_knowledge.py`.

L'ancien RAG renvoyait réellement « Taches jaunes sur feuilles de maïs » pour
« Comment cultiver le maïs ? », « Des termites attaquent mon maïs » et
« Je veux vendre mon maïs ». Pour « termites … pas des chenilles », il retenait
« Limiter les dégâts de la chenille légionnaire sur le maïs ».

La cause était une décision lexicale, avec plusieurs chemins qui contournaient
le contrôle initial. Aucun modèle ne vérifiait que la réponse de la fiche
satisfaisait l'objectif complet.

## Ancien pipeline précisément identifié

1. `/api/v2/analyze`, `/api/v2/query` et `/api/v2/assistant/query` appellent
   `_run_v2_pipeline`. Les routes historiques appellent `resolve_knowledge_answer`.
2. La question complète était conservée, éventuellement traduite en français
   et enrichie de la question utilisateur précédente. Ce n'était donc pas une
   simple perte du texte à l'entrée.
3. `_normalize_free_text` normalise casse, accents et espaces. `_tokenize`
   simplifie les pluriels ; `_rural_tokens` ajoute quelques équivalences et
   retire les mots fréquents. Ces traitements produisent des ensembles de mots,
   sans représentation de l'objectif, de la négation ou du stade.
4. `_find_studio_knowledge_match` consultait jusqu'à 1 000 fiches validées du
   domaine, classées par date. Score brut : titre × 4, tags × 3,5, question × 2,
   réponse × 0,25 ; bonus de sous-chaîne du titre : +15. Le meilleur résultat
   était sélectionné dès `SONGRA_KNOWLEDGE_MIN_SCORE=4`.
5. `_resource_relevant` contrôlait le sujet et surtout maladie contre technique,
   puis autorisait deux mots communs, voire un seul pour une requête d'un mot.
   Il ne distinguait pas suffisamment stockage, vente et ravageurs différents.
6. `retrieve_knowledge` classait les fiches RAG, tickets résolus et fiches Studio
   par titre × 3, tags × 2,5, question × 2, réponse × 0,3. Bonus de sujet +8 et
   de problème +4,5 lorsque fournis. Il n'y avait pas de seuil sémantique.
7. Surtout, si ce classement ne trouvait rien, une recherche par **une seule
   sous-chaîne** réintroduisait les fiches avec un score de 1, sans repasser les
   contrôles de pertinence. Les égalités conservaient l'ordre des résultats SQL.
8. Le résolveur demandait jusqu'à huit résultats (`limit=5`, puis `limit+3`),
   utilisait la première réponse directement en V2 (`allow_external=False`),
   ou la donnait comme base à une reformulation sur les anciennes routes.
9. Les cours étaient choisis dès deux mots communs, avec un contrôle lexical
   supplémentaire pour les questions de calendrier.
10. Après le repli IA, V2 relançait une recherche Studio sur la question **et les
    causes/actions générées**, permettant de remplacer l'analyse par une fiche
    qui n'avait pas répondu à la question initiale.
11. Une autre sélection d'audio humain utilisait un score lexical minimal de 4
    calculé notamment sur la réponse IA. Yingr-AI retournait ses meilleurs
    résultats même de score nul, et élargissait parfois à un autre domaine.
12. Le cache mobile présentait un recouvrement lexical de 80 % comme une
    « fiabilité », puis sélectionnait le premier résultat ou une consultation
    similaire. Ce ratio n'était pas une probabilité de réponse correcte.
13. Le repli IA n'arrivait qu'après ces sélections. Une fausse correspondance
    suffisait donc à empêcher ce repli.

## Nouvelle sélection

`knowledge_relevance.py` comprend la question complète avec le fournisseur
configuré, avant le classement des candidats : domaine, intention, culture,
animal, problème, symptômes, action demandée, objectif et contexte. Il conserve
simultanément `originalQuery` et `normalizedQuery`. Les concepts et synonymes
proposés enrichissent la recherche ; ils ne remplacent jamais la question.

Le rappel reste lexical, enrichi sémantiquement. **Aucun index vectoriel ni score
de cosine n'a été ajouté.** Son score ne peut plus autoriser une réponse. Au plus
cinq fiches et cinq cours sont transmis à une validation sémantique commune.
Les filtres de statut, domaine et organisation des cours sont conservés.

Le validateur lit le titre, la question ET le contenu effectivement affiché. Il
évalue chaque candidat sur six critères : sujet, intention, problème, contexte,
objectif et réponse effective à la question. Une fiche doit satisfaire tous les
critères, avoir un score dans une bande de réponse directe et fournir une citation
présente dans sa réponse. Les IDs inconnus, doublons, résultats incomplets,
booléens invalides ou erreurs fournisseur ne peuvent autoriser une fiche.

Les candidats acceptés sont classés par leur score sémantique. Le résolveur
privilégie Studio, puis RAG validé, puis cours pertinent. Sinon le pipeline
appelle son IA générale avec la question complète. Une fiche rejetée en texte
ne peut plus être réintroduite par les recommandations générées par cette IA.

Pour une photo, l'observation visuelle passe d'abord. Le rapprochement utilise
la question et les observations/diagnostics avec leur incertitude, pas les
actions inventoriées par le modèle. Si l'analyse exige une précision, aucune
fiche spécifique n'est imposée.

Si plusieurs solutions concurrentes restent indécidables, la réponse contient
une question complémentaire ; aucune fiche, action ni consommation de quota
d'analyse n'est imposée. Les doublons équivalents ne nécessitent pas une
clarification simplement parce qu'ils ont des scores égaux.

Les routes Yingr-AI utilisent le même validateur. L'audio est attaché à la fiche
déjà validée ; une recherche sur la réponse générée ne peut plus ressusciter une
fiche rejetée. Les anciens helpers de réutilisation approchée passent également
par le contrôle commun.

Hors ligne, aucune proximité lexicale ne choisit une nouvelle fiche. Le cache
de connaissances exige une validation serveur associée à la question complète
identique. Une consultation peut être reprise pour la question identique, sans
photo ni nouveau contexte, et seulement si les réponses sauvegardées ne se
contredisent pas. Les fiches et consultations restent disponibles dans le stockage.

## Score et seuil, avec mesures

`SONGRA_SEMANTIC_MIN_SCORE=3`, configurable à **3 ou 4 uniquement** :

| Score ordinal | Sens | Utilisation principale |
|---|---|---|
| 0 | Hors sujet / contradiction | Rejet |
| 1 | Même sujet / mots communs | Rejet |
| 2 | Complément ou réponse partielle, objectif essentiel absent | Rejet |
| 3 | Réponse directe, détails secondaires absents | Validation des six critères obligatoire |
| 4 | Réponse directe complète, contexte spécifique compatible | Même validation obligatoire |

Ces scores sont des **classes ordinales définies par une grille**, pas des
probabilités calibrées. Un score de 4 avec un seul critère faux ou une citation
absente est rejeté. Les anciens seuils lexicaux 4 et 2 n'interviennent plus.

Calibration avec Groq `qwen/qwen3.8-27b` : dix scénarios annotés sur neuf fiches
de test, puis sept recherches sur les dix fiches agricoles réellement présentes.
Les anciens scores lexicaux observés vont de 0 à 25,4. Le maximum était 7,8
pour culture, termites et vente ; 16,2 pour la requête avec négation ; 9,8 pour
l'assurance contre la grêle. Ces valeurs ne suffisaient donc pas à distinguer
une réponse complète d'une mauvaise correspondance.

Dans la première calibration complète, les 50 évaluations étaient réparties
ainsi : 39 scores 0, 3 scores 1, 2 scores 2, 1 score 3 et 5 scores 4. Sur les
étiquettes attendues, le seuil 3 donne zéro faux positif et un rejet conservateur ;
le seuil 4 donne zéro faux positif et deux rejets. Le cas positif « termites »
est noté 3 : exiger 4 écarterait ce conseil direct malgré sa pertinence. Cela
justifie 3 plutôt que 4 ; abaisser à 2 autoriserait explicitement une réponse
partielle, ce qui est interdit.

Limite constatée et conservée dans les résultats : la fiche de test « termites »
donne inspection et orientation vers un conseiller, sans méthode de lutte
complète. Pour la requête avec négation, le modèle oscille entre 2 et 3. La
consigne rappelle de respecter la négation, mais ne contourne pas le critère
de réponse à l'objectif. Une vérification ciblée a accepté la fiche à 3 ; la
suivante l'a rejetée à 2 pour réponse insuffisante. Dans tous ces essais la
fiche « chenilles » est rejetée. **Aucun seuil n'a été abaissé pour forcer une
sélection.** Le repli est préférable à une validation incertaine.

Il s'agit d'une petite calibration locale, pas d'une garantie de précision sur
la base de production ni d'une validation agronomique des contenus existants.
La distribution des scores et toutes les décisions sont conservées dans
`tests/relevance_calibration.json`, y compris les réévaluations divergentes.

## Avant / après sur la même base locale

| Question | Ancien résolveur rejoué | Après contrôle sémantique |
|---|---|---|
| Cultiver le maïs | Fiche jaunissement #1 | Rejet des réponses partielles ; repli faute de formation complète |
| Termites dans le maïs | Fiche jaunissement #1 | Rejet ; repli faute de fiche termites |
| Conserver après récolte | Conservation #25 | Conservation #25, score 4 |
| Feuilles de maïs jaunes | Jaunissement #1 | Fiche #1, score 3, présentant carence **ou** drainage ; pas de maladie unique confirmée |
| Vendre le maïs | Fiche jaunissement #1 | Rejet ; repli faute de ressource de commercialisation |
| Termites, pas chenilles | Chenille légionnaire #24 | Rejet de #24 ; repli |
| Assurance contre la grêle | Chenille légionnaire #24 | Rejet ; repli |

Dans les scénarios contrôlés disposant de fiches complètes, « Cultiver le maïs »,
« Termites du maïs », « Conservation », « Commercialisation » et « Culture de
l'ail » sont sélectionnés pour leurs intentions respectives. « Maladies
fongiques de l'ail » est rejeté pour une demande de culture ; « Pourriture des
tiges » est rejetée pour une demande de stockage ; la carence en azote confirmée
n'est pas imposée pour un simple jaunissement ou après de fortes pluies.

## Fichiers modifiés

- `main.py` : sélection commune, suppression des décisions lexicales et des
  contournements, clarification, recherche V2 exécutée sans bloquer sa boucle async.
- `knowledge_relevance.py` : compréhension, validation, preuve textuelle, gestion
  des réponses invalides, score ordinal et journal d'audit.
- `yingr_ai_services.py`, `yingr_ai_api.py` : même contrôle pour les routes V3,
  absence d'élargissement automatique à un autre domaine.
- `v2_services.py`, `.env.example` : modèle Groq par défaut déjà vérifié disponible,
  configuration du seuil sémantique. Les secrets ne sont pas versionnés.
- `tests/test_knowledge_relevance.py`, `tests/conftest.py` : tests de contrat et
  d'intégration avec fournisseur contrôlé ; aucun accès réseau dans pytest.
- `tests/test_rural_multimodal_pipeline.py`, `tests/test_agricultural_assistant_scope.py` :
  injection du nouveau fournisseur contrôlé dans les tests existants.
- `tests/calibrate_knowledge_relevance.py`, `tests/replay_legacy_knowledge.py` et
  leurs rapports JSON : calibration réelle et reproduction de l'ancien défaut.
- `songra_app/lib/services/offline_service.dart`,
  `songra_app/test/offline_relevance_test.dart` : suppression des réponses
  approximatives hors ligne et tests du cache.

## Journaux et vérifications

`[SONGRA-RELEVANCE]`, `[YINGR-RELEVANCE]` et `[SONGRA-REUSE-RELEVANCE] enregistrent
question complète et normalisée, domaine, intention, culture, animal, problème,
symptômes, objectif, IDs/titres/scores des candidats, décision de chaque validation,
preuve, motif de rejet, ID/score choisi et repli. Le journal V2 confirme le
repli réellement déclenché et le fournisseur utilisé.

Tests automatiques : intentions A/B/C/E différentes sur le même maïs ; rejet F ;
repli G ; jaunissement sans maladie arbitraire D ; ail formation contre maladie ;
cours pertinent et organisation ; panne du validateur ; absence de sous-chaîne
de secours ; six critères obligatoires malgré un score lexical élevé ; JSON
compact/invalide ; IDs manquants/dupliqués/inconnus ; preuve absente ; seuils 3/4 ;
ambiguïté sans facturation ; alias health/urgence ; V3 et cache mobile.

Résultats finaux : **83 tests backend réussis, 13 tests Flutter réussis**, aucune
erreur ni avertissement dans l'analyse ciblée du fichier mobile et de ses tests.
La calibration fournisseur initiale satisfait 9/10 étiquettes de sélection ;
le dixième cas rejette prudemment une réponse partielle plutôt que de choisir
la fiche chenilles. Les sept recherches sur le corpus réel confirment les
sélections/rejets du tableau. Les tests automatisés contrôlent séparément que
ces rejets déclenchent l'IA générale, et que l'ambiguïté déclenche une précision.

Commandes :

```text
cd backend
python -m pytest tests -q
python tests/replay_legacy_knowledge.py
python tests/calibrate_knowledge_relevance.py --output tests/relevance_calibration.json
python tests/calibrate_knowledge_relevance.py --output tests/relevance_calibration.json --case F
python tests/calibrate_knowledge_relevance.py --output tests/relevance_calibration.json --real-only

cd ../songra_app
flutter test --no-pub
flutter analyze --no-pub --no-fatal-infos --no-fatal-warnings lib/services/offline_service.dart test/offline_relevance_test.dart
```

Les appels de calibration réelle sont espacés en cas de limite de débit. Le
contrat compact limite leur sortie à 1 000 tokens, compatible avec la limite
observée sur ce compte Groq. En production, une indisponibilité du validateur
rejette les candidats et permet le repli ; elle ne justifie jamais un retour
au premier résultat lexical.

La modification backend doit être installée puis le service redémarré sur la
VM. La modification du cache mobile nécessite une nouvelle version de l'app.
La signature de la version déjà publiée doit conserver son keystore existant.
