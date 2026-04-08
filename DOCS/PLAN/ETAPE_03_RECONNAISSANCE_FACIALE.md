# 🧑 ÉTAPE 3 — Reconnaissance Faciale et Gestion de la Whitelist

> **Durée estimée : 3-4 jours**
> **Priorité : HAUTE**
> **Prérequis : ÉTAPE 2 complétée et validée**

---

## 🎯 Objectif

Implémenter la reconnaissance faciale pour identifier les personnes à partir d'une whitelist (liste blanche). Le système doit pouvoir distinguer les personnes autorisées des inconnus en temps réel, et déclencher les événements appropriés.

---

## 📋 Tâches détaillées

### 3.1 — Module Face Manager (`src/core/face_manager.py`)

**Architecture du module :**

```python
class FaceManager:
    """
    Gestion complète de la reconnaissance faciale :
    - Détection de visages
    - Extraction d'embeddings (InsightFace)
    - Comparaison avec la whitelist
    - Enrôlement de nouvelles personnes
    """
```

**Technologies :**
- **InsightFace** (`insightface`) pour l'extraction d'embeddings (vecteurs 512D).
- **ONNX Runtime** pour l'inférence optimisée.
- **SciPy** pour le calcul de similarité cosinus.

### 3.2 — Gestion de la Whitelist

**Structure de stockage :**
```
data/whitelist/
├── registry.json          # Registre des personnes
├── embeddings/
│   ├── ilyas_001.npy      # Embedding vectoriel
│   ├── ilyas_002.npy      # Plusieurs angles par personne
│   └── marie_001.npy
└── photos/
    ├── ilyas_ref_1.jpg    # Photo de référence
    ├── ilyas_ref_2.jpg
    └── marie_ref_1.jpg
```

**Format `registry.json` :**
```json
{
  "persons": [
    {
      "id": "person_001",
      "name": "Ilyas Ghandaoui",
      "role": "Propriétaire",
      "access_level": "admin",
      "embeddings": ["ilyas_001.npy", "ilyas_002.npy", "ilyas_003.npy"],
      "photos": ["ilyas_ref_1.jpg", "ilyas_ref_2.jpg"],
      "enrolled_at": "2026-04-08T12:00:00Z",
      "last_seen": null,
      "notes": "Propriétaire principal"
    }
  ]
}
```

### 3.3 — Fonctionnalités d'enrôlement

Implémenter un processus d'enrôlement complet :

1. **Enrôlement par photo** : L'utilisateur fournit 3-5 photos de la personne.
2. **Enrôlement par caméra** : Capture de 5 angles différents en direct (face, profil gauche, profil droit, etc.).
3. **Validation de qualité** : Vérifier que le visage est bien visible, bien éclairé, et net.
4. **Multi-embedding** : Stocker plusieurs embeddings par personne pour améliorer la robustesse.

### 3.4 — Pipeline de reconnaissance

```
Frame → Détection YOLO (personnes) → Crop régions visage 
  → InsightFace (embedding) → Comparaison whitelist 
  → [CONNU: nom + rôle] ou [INCONNU: alerte]
```

**Détails techniques :**
- Extraction de visage avec marge de 20% autour du bbox YOLO.
- Embedding 512 dimensions via InsightFace.
- Comparaison par **similarité cosinus** avec tous les embeddings de la whitelist.
- Seuil configurable (défaut: 0.6, RÈGLE : > 0.6 = match, 0.4-0.6 = incertain, < 0.4 = inconnu).
- **Cache d'embeddings** : Ne pas recalculer l'embedding si le visage n'a pas bougé significativement.

### 3.5 — Intégration avec le Tracker

Lier la reconnaissance faciale au module de tracking (étape 2) :

```python
class TrackedEntity:
    # Ajouter à la dataclass existante:
    face_id: Optional[str] = None        # ID de la personne si reconnue
    face_name: Optional[str] = None      # Nom de la personne
    face_confidence: float = 0.0         # Score de reconnaissance
    face_status: str = "unknown"         # "known", "unknown", "uncertain"
```

- Une fois qu'un visage est reconnu, le lier au `track_id` pour ne pas recalculer à chaque frame.
- Recalculer l'embedding toutes les **30 frames** ou si le visage change d'angle significativement.

### 3.6 — Gestion des inconnus

Quand un visage inconnu est détecté :
1. **Snapshot automatique** : Sauvegarder un crop du visage dans `data/snapshots/unknowns/`.
2. **Événement `face_unknown`** : Émettre avec le snapshot et les métadonnées.
3. **Historique** : Logger l'heure, la durée de présence, et le comportement.

**Événements émis :**
- `face_recognized` : `{person_id, name, confidence, bbox}`
- `face_unknown` : `{bbox, snapshot_path, track_id}`
- `face_uncertain` : `{bbox, best_match, confidence}`

---

## 🧪 Tests de validation (Étape 3)

| # | Test | Description | Résultat attendu |
|---|------|-------------|-------------------|
| 1 | Enroll Photo | Enrôler avec 3 photos | Embeddings sauvés |
| 2 | Enroll Camera | Capture 5 angles | 5 embeddings créés |
| 3 | Recognize Known | Personne enrôlée devant caméra | Nom affiché + score > 0.6 |
| 4 | Recognize Unknown | Personne NON enrôlée | Label "INCONNU" + snapshot |
| 5 | Multi-person | 2 personnes (1 connue, 1 inconnue) | Distinction correcte |
| 6 | Tracker Link | ID tracker lié au face_id | Cohérence maintenue |
| 7 | Performance | Temps de reconnaissance | < 100ms par visage |
| 8 | Registry CRUD | Ajouter/Supprimer personne | registry.json mis à jour |

### Tests unitaires obligatoires :
- `tests/test_face_manager.py` — Tests avec images de test.
- `tests/test_whitelist.py` — CRUD sur le registre.

---

## 📦 Livrables de l'étape

- [ ] `src/core/face_manager.py` — Module complet
- [ ] `data/whitelist/registry.json` — Registre initial
- [ ] Pipeline de reconnaissance intégré au tracking
- [ ] Snapshots automatiques des inconnus
- [ ] Script d'enrôlement : `scripts/enroll_person.py`
- [ ] Demo : `scripts/demo_face_recognition.py`
- [ ] Tests unitaires
- [ ] `RESUME_ETAPE_03.md`

---

## ⚠️ Points d'attention

- **InsightFace** nécessite le téléchargement d'un modèle (~300MB). Le télécharger au premier lancement.
- **Qualité des photos d'enrôlement** : Des photos floues ou mal éclairées donneront de mauvais résultats.
- **Vie privée** : Les embeddings sont des données biométriques sensibles. Les stocker de manière sécurisée.
- **Ne PAS implémenter l'UI d'enrôlement** — seulement le script CLI. L'UI viendra à l'étape 6.

---

## 🔗 Dépendances pour l'étape suivante

L'étape 4 nécessite :
- ✅ `face_manager.py` capable de distinguer connus/inconnus
- ✅ Trackers enrichis avec `face_id` et `face_name`
- ✅ Événements de reconnaissance émis correctement
- ✅ Au moins 1 personne enrôlée pour les tests
