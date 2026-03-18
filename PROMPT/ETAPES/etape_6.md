# 🚀 Étape 6 : Améliorations Avancées de l'IA (Précision & Latence)

Ce guide détaille la procédure pour implémenter les optimisations d'IA suite à l'analyse du système, afin de le rendre plus interactif, rapide, et précis.

## 🎯 Objectifs de l'Étape
1. **Contexte Visuel (Vision)** : Améliorer la description de la scène par l'IA Guard en intégrant systématiquement l'analyse visuelle de Moondream pour comprendre le contexte global au lieu de se baser uniquement sur des règles géométriques.
2. **Précision ST-GCN** : Passer l'entrée du modèle de 2 canaux `(x, y)` à 3 canaux `(x, y, confidence)`.
3. **Réduction Latence AI Guard** : Rendre l'Agent conversationnel ultra-rapide avec `faster-whisper` et le streaming Ollama + TTS.

---

## 🛠 6.1. Amélioration du Contexte et Description de la Scène
Dans cette sous-étape, nous allons retirer les heuristiques codées en dur pour la détection d'objets (comme "téléphone" via distance main-oreille) au profit de l'analyseur Vision intégré au Guard.

**Modifications à faire :**
1. **`src/pipeline/analyzer.py`** :
   - Supprimer ou désactiver la méthode `_detect_objects_by_pose`.
   - Augmenter si besoin la fréquence de l'analyse YOLO26n locale pour la détection pure d'objets.
2. **`src/behavior/ai_guard.py`** :
   - Modifier le `SYSTEM_PROMPT` pour exiger que l'agent se repose fortement sur l'analyse visuelle de la scène qu'il reçoit.
   - Ajuster la fonction `_analyze_frame_vision` pour s'assurer que Moondream annote spécifiquement les "objets portés dans les mains" de manière robuste.

---

## 🛠 6.2. Amélioration du Modèle ST-GCN (Passage à C=3)
Cette modification permet au réseau de comprendre quand une articulation n'est pas visible (ex: jambes coupées par le bas de la caméra), évitant qu'il n'interprète `x=0, y=0` comme une vraie position repliée.

**Modifications à faire :**
1. **`src/config.py`** :
   - Passer `STGCN_IN_CHANNELS` à `3`.
2. **`src/pipeline/detector.py`** :
   - Mettre à jour `PersonDetection` pour utiliser les keypoints bruts `(17, 3)` retournés par YOLO plutôt que de les tronquer à `(17, 2)` dans `keypoints_xy`.
3. **`src/behavior/skeleton_buffer.py`** :
   - Assurer que le buffer temporel accumule bien des matrices de taille `(17, 3)`.
4. **`src/models/stgcn/model.py`** :
   - Le modèle acceptera automatiquement `in_channels=3`.
   - *Attention* : Les poids entraînés pour C=2 (si présents) seront incompatibles. Il faudra prévoir une étape de ré-entrainement ou ajouter un petit pont adaptateur Conv1x1 (de 3 vers 2) en attendant le ré-entraînement.

---

## 🛠 6.3. Réduction de la Latence de l'Agent Conversationnel
Passer d'une latence conversationnelle de ~4s à ~1s.

**Modifications à faire (`src/behavior/ai_guard.py`) :**
1. **STT Rapide** : 
   - Installer `faster-whisper`.
   - Remplacer l'instance de `whisper` par `faster_whisper.WhisperModel`. Cette version s'exécute beaucoup plus rapidement grâce à CTranslate2.
2. **Streaming du LLM (Ollama)** : 
   - Modifier l'appel API `requests.post` vers Ollama : `"stream": True`.
   - Ajouter un itérateur pour lire la réponse du LLM token par token.
3. **Chunking et TTS Asynchrone** :
   - Accumuler les tokens jusqu'à rencontrer un délimiteur de fin de phrase (`.`, `!`, `?` ou `,`).
   - Envoyer immédiatement cette phrase à `edge-tts` (ou un nouveau TTS local rapide comme `Piper`) pour la synthèse et la lecture audio, **pendant** que le LLM continue de générer la phrase suivante.

---

## ✅ Critères de Validation
- [ ] Une conversation vocale naturelle avec l'AI Guard a une **latence perçue < 1.5s** avant le premier mot vocal de la réponse.
- [ ] Le ST-GCN accepte l'entrée `(1, 3, 30, 17)`. Si on masque le bas du corps (confidence 0), les fausses détections de chutes/actions anormales diminuent drastiquement.
- [ ] Dans le log de l'AI Guard, on voit la description de "Moondream" fournir explicitement le contexte des objets (ex: "tient un téléphone portable"). Les heuristiques de l'analyzer sont supprimées.

> [!TIP]
> **Prochaine action** : Ouvrir `src/behavior/ai_guard.py` et commencer par 6.1 et 6.3, qui sont les améliorations les plus impactantes visuellement sans nécessiter de ré-entraîner de modèle Pytorch.
