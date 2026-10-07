# Bibliothèque pédagogique hors connexion — MVP

## Réutilisation et comportement

Le MVP réutilise `AcademyCourseDB`, les identifiants et étapes existants, les
statuts publié/brouillon, les enregistrements par langue et le contrôle d'accès
Académie. Aucun cours ni traduction n'est créé par le téléchargement. Les cours
déjà présents restent en base ; le démarrage du serveur ne publie plus de
modèles de cours automatiquement.

L'application conserve ses fiches rapides, consultations et synchronisations.
Les entrées Apprendre et Hors connexion conduisent respectivement au catalogue
et à la bibliothèque visuelle locale. Les recommandations de l'assistant
continuent à suivre sa réponse, avec accès direct au cours téléchargé et un
bouton de téléchargement sinon. Le sujet interprété par le backend est conservé
dans la réponse et l'historique pour les questions en langue locale.

## API et médias

`GET /api/academy/courses/{id}/offline-info` vérifie la taille et la version
avant confirmation, sans exposer les leçons ou URL protégées ni ouvrir d'accès.
Annuler à ce stade ne consomme aucun quota. Après accord, l'application utilise
le contrôle d'accès existant puis le manifeste ; une version modifiée entre
ces étapes impose une nouvelle tentative.

`GET /api/academy/courses/{id}/offline-manifest` exige un JWT utilisateur,
le bon périmètre d'organisation et un accès au cours déjà ouvert. Il ne crée
aucun droit ni événement de quota. Il fournit les vraies données, une version
SHA-256 du contenu, les médias uniques, leurs empreintes et tailles quand les
fichiers existent sur le serveur. Un média local absent bloque le manifeste ;
une taille externe inconnue reste inconnue. L'interface affiche la taille des
**médias**, sans prétendre connaître les octets d'un package externe inconnu.

Les nouveaux uploads d'illustrations sont limités à 20 Mo, orientés selon EXIF,
réduits à 1440 pixels maximum et compressés en WebP, sans ajout de contenu
pédagogique. Les audios humains sont conservés dans leur format d'origine :
MP3/M4A/AAC/Ogg sont utilisables directement. Les anciens fichiers lourds ne
sont pas recompressés automatiquement ; leur optimisation éditoriale reste à
faire si nécessaire. Aucune dépendance FFmpeg n'est imposée à la VM.

Les langues utilisent un identifiant extensible. Une langue inconnue n'est
plus rebaptisée « français » et une date d'upload absente n'est pas inventée,
ce qui garantit aussi des versions stables pour les anciens audios.

## Stockage et interruption

`AcademyOfflineService` utilise le répertoire durable `path_provider`, séparé
par compte, et des index/progressions SharedPreferences. Les médias sont
identifiés et vérifiés par SHA-256. Ils ne passent pas par le cache temporaire
d'uploads qui peut être nettoyé automatiquement.

L'instantané local contient les données réelles du cours, sa version, la date
de téléchargement et les chemins/taille/empreinte des médias. L'index actif
n'est publié qu'après téléchargement complet. La lecture vérifie le JSON et
tous les fichiers ; un fichier manquant ou corrompu retire la disponibilité.
Un téléchargement interrompu garde la version antérieure et les médias déjà
terminés réutilisables ; les fichiers partiels ne sont pas déclarés disponibles.
La reprise se fait par nouvelle tentative avec réutilisation des fichiers
complets, pas par requêtes HTTP Range.

Les anciens caches Académie sont conservés et migrés une fois vers le compte
enregistré avant la mise à jour. Un cache sans propriétaire reste conservé,
sans être attribué à un nouveau compte. Les écritures de l'index sont sérialisées
pour préserver les téléchargements pendant une vérification des mises à jour.

Les packs regroupent les vrais cours du catalogue par culture. Un téléchargement
partiel ne produit pas le statut « Pack disponible ». Les médias partagés sont
comptés une fois et réutilisés. Les cours complètement téléchargés restent
ouverts même si une autre formation du pack échoue.

## Lecteur et progression

`VisualCourseScreen` affiche les étapes réelles, sans imposer un nombre fixe :
grande illustration, titre, texte secondaire dépliable, gros boutons et
progression. Le lecteur téléchargé utilise exclusivement `Image.file` et
`DeviceFileSource`. Aucune traduction/TTS ni requête serveur n'est nécessaire
pour parcourir le cours. Le bouton audio et le choix de langue apparaissent
uniquement pour les enregistrements réellement fournis et téléchargés.

La progression stocke les identifiants des étapes terminées, l'étape courante
et une date réelle. Les anciens identifiants supprimés ne faussent pas la
progression après une mise à jour. Le backend n'a pas d'API de progression
Académie existante : la progression reste locale dans ce MVP.

À la reconnexion, le catalogue permet de détecter les nouvelles versions. Une
mise à jour est proposée, sans téléchargement ni remplacement automatique.
Les suggestions déjà validées sont conservées pour la question complète et
le compte ; seules celles dont le cours est téléchargé sont réutilisées hors
ligne. Aucun rapprochement par un simple mot ne suggère un nouveau cours.

## Validation et limites de terrain

Validation automatisée : 103 tests backend et 31 tests Flutter réussis.
L'analyse ciblée des 12 fichiers Dart concernés ne signale aucune erreur.

Les tests couvrent fichiers images/audio conservés après recréation du service,
transport réseau entièrement désactivé, bibliothèque → catégorie → culture →
cours, images locales, audio depuis un fichier local, étapes et progression,
fermeture/reprise, annulation, fichier corrompu, téléchargement interrompu,
mise à jour interrompue, réutilisation des médias, isolation des comptes,
absence de langue inventée, taille/déduplication et statut honnête des packs.
Ils utilisent des contenus explicitement marqués TEST, jamais publiés dans
l'Académie. Le lecteur audio est simulé dans les tests Flutter ; l'écoute
effective sur un téléphone reste un contrôle matériel distinct.

La base **locale** auditée contient zéro cours. Cela ne signifie pas que la VM
n'en contient pas : `https://songraback.yingr-ai.com/health` répond 200, mais
`/api/academy/courses` répond 401 sans compte et le nom SSH `finavi` n'est pas
résolu depuis ce poste. La connexion SSH au domaine atteint le serveur mais
est refusée (`Permission denied (publickey)`). Une sortie en lecture seule listant identifiants,
titres, nombre d'étapes/images/audios a été demandée pour sélectionner le vrai
cours de référence. Aucun contenu de production n'a été inventé ou copié.

Après déploiement du backend et d'une application compatible : choisir un cours
réel publié dont les étapes ont leurs illustrations/enregistrements validés,
l'ouvrir, télécharger, activer le mode avion, relancer, ouvrir Hors connexion,
parcourir toutes les étapes, écouter, avancer, fermer et reprendre. Tester aussi
une coupure pendant le téléchargement. Les fonctionnalités sont prêtes pour
cette vérification ; l'acceptation sur un cours réel de la VM n'est pas déclarée
acquise sans cette vérification.
