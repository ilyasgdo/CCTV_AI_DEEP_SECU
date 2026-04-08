# 🌟 ÉTAPE 9 — Fonctionnalités Avancées (Post-MVP)

> **Durée estimée : 5-7 jours**
> **Priorité : MOYENNE**
> **Prérequis : ÉTAPE 8 complétée et validée**

---

## 🎯 Objectif

Ajouter les fonctionnalités avancées qui différencient Sentinel-AI d'une simple caméra de surveillance. Ces fonctionnalités augmentent la valeur du produit pour les investisseurs et les utilisateurs finaux. Elles ne sont PAS nécessaires pour le MVP mais sont prévues pour la version 1.1+.

---

## 📋 Tâches détaillées

### 9.1 — Enregistrement de clips d'alerte

Quand une alerte est déclenchée, sauvegarder automatiquement un clip vidéo :

```python
class AlertClipRecorder:
    """
    Enregistre un clip vidéo horodaté lors d'une alerte.
    Buffer circulaire : garde les N secondes AVANT l'alerte + N secondes APRÈS.
    """
    
    def __init__(self, pre_alert_seconds: int = 10, post_alert_seconds: int = 15):
        self.buffer = CircularBuffer(max_seconds=pre_alert_seconds)
    
    async def on_alert(self, alert_data: dict):
        """
        1. Prendre les frames du buffer (10s avant)
        2. Continuer à enregistrer 15s après
        3. Compiler en fichier MP4
        4. Sauvegarder dans data/clips/
        """
```

**Format fichier :** `data/clips/ALERT_2026-04-08_12-05-30_intrusion.mp4`

### 9.2 — Rapports PDF automatiques

Générer des rapports PDF journaliers/hebdomadaires :

```python
class ReportGenerator:
    """
    Génère des rapports PDF avec statistiques et graphiques.
    Utilise fpdf2 + matplotlib.
    """
    
    def generate_daily_report(self, date: datetime) -> str:
        """
        Contenu du rapport:
        - Résumé de la journée
        - Nombre de personnes détectées
        - Temps de présence moyen
        - Liste des alertes (avec screenshots)
        - Graphiques : activité par heure, types d'événements
        - Personnes identifiées vs inconnus
        """
```

### 9.3 — Support multi-caméras

Permettre la surveillance de plusieurs caméras simultanément :

```python
class MultiCameraManager:
    """
    Gère N caméras en parallèle.
    Chaque caméra a son propre pipeline de perception.
    Les analyses LLM sont partagées (une seule requête groupée).
    """
    
    def __init__(self, camera_configs: List[CameraConfig]):
        self.cameras = [
            CameraPipeline(config) for config in camera_configs
        ]
```

**Dashboard multi-caméra :**
```
┌─────────────────────────────────────┐
│  Mosaïque 2x2                       │
│  ┌──────────┐  ┌──────────┐        │
│  │ Cam 1    │  │ Cam 2    │        │
│  │ (Entrée) │  │ (Parking)│        │
│  └──────────┘  └──────────┘        │
│  ┌──────────┐  ┌──────────┐        │
│  │ Cam 3    │  │ Cam 4    │        │
│  │ (Jardin) │  │ (Bureau) │        │
│  └──────────┘  └──────────┘        │
└─────────────────────────────────────┘
```

### 9.4 — Zones de surveillance configurables

Permettre de dessiner des zones d'intérêt sur l'image :

- **Zone d'intrusion** : Déclencher une alerte si quelqu'un entre dans la zone.
- **Zone de passage** : Compter les entrées/sorties.
- **Zone d'exclusion** : Ignorer la détection dans cette zone.

**Interface :** Dessin de polygones directement sur le flux vidéo dans le dashboard.

### 9.5 — Détection d'anomalies comportementales

Aller au-delà de la simple détection d'objets :

- **Rôdeur** : Personne qui reste dans la zone > X minutes sans objectif clair.
- **Mouvement rapide** : Course, fuite.
- **Chute** : Détection de chute d'une personne (pose estimation).
- **Objets abandonnés** : Objet statique apparaissant soudainement.
- **Comportement nocturne** : Activité anormale pendant les heures de nuit.

### 9.6 — Notifications multi-canal

Étendre le système de notifications :

| Canal | Technologie | Priorité |
|-------|-------------|----------|
| Email | SMTP | Toutes alertes |
| SMS | Twilio API | Alertes critiques |
| Telegram | Bot API | Temps réel + photos |
| Push Web | Service Worker | Dashboard |
| Webhook | HTTP POST | Intégration externe |

### 9.7 — API externe / Intégrations

Exposer une API REST publique (avec auth) pour permettre l'intégration :

```
POST /api/v1/webhook/alert     → Recevoir les alertes
GET  /api/v1/cameras           → Lister les caméras
GET  /api/v1/stream/{cam_id}   → Flux vidéo d'une caméra
GET  /api/v1/report/daily      → Récupérer le rapport du jour
```

**Intégrations possibles :**
- Home Assistant / Domotique
- Serrures connectées (ouvrir/fermer)
- Systèmes d'alarme existants
- Plateformes cloud (AWS, GCP)

---

## 🧪 Tests de validation (Étape 9)

| # | Test | Résultat attendu |
|---|------|-------------------|
| 1 | Clip enregistrement | Clip MP4 sauvegardé lors d'une alerte |
| 2 | Rapport PDF | PDF généré avec graphiques lisibles |
| 3 | Multi-caméra | 2 flux simultanés sans lag majeur |
| 4 | Zone intrusion | Alerte quand zone franchie |
| 5 | Détection rôdeur | Alerte après 5 min de présence |
| 6 | Notification Telegram | Message + photo reçus |
| 7 | API externe | Endpoints accessibles avec auth |

---

## 📦 Livrables de l'étape

- [ ] Enregistrement de clips d'alerte
- [ ] Génération de rapports PDF
- [ ] Support multi-caméras (2-4)
- [ ] Zones de surveillance configurables
- [ ] Détection d'anomalies (base)
- [ ] Au moins 1 notification supplémentaire (Telegram ou SMS)
- [ ] Documentation des nouvelles fonctionnalités
- [ ] `RESUME_ETAPE_09.md`

---

## ⚠️ Points d'attention

- **Multi-caméra** : Attention à la charge CPU. 4 caméras = 4x le processing.
- **Clips vidéo** : L'encodage MP4 est gourmand — le faire dans un thread séparé.
- **APIs externes** (Twilio, Telegram) nécessitent des clés API — `.env`.
- **Ne PAS sur-implémenter** — choisir 3-4 features de cette liste pour le sprint.

---

## 🔗 Dépendances pour l'étape suivante

L'étape 10 nécessite :
- ✅ Fonctionnalités avancées choisies et implémentées
- ✅ Tests validés
- ✅ Documentation mise à jour
