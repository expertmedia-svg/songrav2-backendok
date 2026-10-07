# Académie : complément après la réponse SONGRA

La réponse principale utilise les fiches validées et le RAG, puis l'analyse IA
si nécessaire. Un résumé de formation ne remplace plus cette réponse.

Le résultat V2 indique `learning_requested`. Les demandes techniques et
d'apprentissage sont éligibles ; les diagnostics de maladie ou d'incident sont
exclus. La pertinence des formations est ensuite contrôlée par le même moteur
sémantique que les connaissances, avec son seuil et ses contrôles obligatoires.

Après le premier affichage, l'application appelle l'endpoint authentifié
`POST /api/academy/recommendations` avec la question complète et son domaine.
La recherche utilise uniquement les cours publiés de `AcademyCourseDB`, dans
le périmètre d'organisation de l'utilisateur. Les titres, résumés, couvertures,
types et identifiants proviennent du catalogue existant. Les cinq candidats
présélectionnés sont validés sémantiquement ; au maximum trois cours acceptés
sont renvoyés, par score décroissant. Aucun cours ni accès n'est créé.

La section « Pour aller plus loin » apparaît en fin de résultat seulement
lorsque des formations sont disponibles. Une panne réseau ou une recherche
sans correspondance laisse la réponse intacte, sans message de catalogue vide.
« Voir la formation » ouvre le cours réel via son identifiant et le parcours
d'accès/paiement existant. Le retour depuis sa fiche revient directement à
SONGRA. Le catalogue actuel contient des cours internes ; le composant sait
ouvrir une URL de formation externe si une source réelle en fournit une.

Les métadonnées d'apprentissage sont conservées dans l'historique et les copies
de réponse. Les suggestions sont recherchées à nouveau à l'ouverture, afin de
respecter le statut et le périmètre actuels des cours.

Validation : 93 tests backend et 16 tests Flutter passent, dont réponse avant
suggestion, correspondance réelle, absence de cours, maximum de trois,
diagnostic exclu, cours non publié/hors organisation, bouton, identifiant
correct et retour à SONGRA. Les appels IA et HTTP sont simulés dans les tests ;
la validation sur VM nécessite le déploiement du backend et de l'application.
L'analyse ciblée des nouveaux composants est également exécutée. Les écrans
préexistants conservent leurs avertissements Dart (champs inutilisés et API
dépréciées), sans erreur de compilation.
