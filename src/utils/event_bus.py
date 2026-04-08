"""
Event Bus pour Sentinel-AI.

Bus d'événements basé sur le pattern Observer/Pub-Sub pour la
communication inter-modules sans couplage direct.

Chaque module peut :
- s'abonner à des événements (subscribe)
- émettre des événements (emit)
- se désabonner (unsubscribe)

Thread-safe : Les événements peuvent être émis depuis n'importe
quel thread.

Usage:
    from src.utils.event_bus import EventBus

    bus = EventBus()

    def on_person(data):
        print(f"Personne détectée: {data}")

    bus.subscribe("person_detected", on_person)
    bus.emit("person_detected", {"bbox": (10, 20, 100, 200)})
    bus.unsubscribe("person_detected", on_person)

Événements prévus:
    "person_detected"     -> {bbox, confidence, frame_id}
    "face_recognized"     -> {name, confidence, bbox}
    "face_unknown"        -> {bbox, snapshot_path}
    "analysis_ready"      -> {snapshot, context}
    "llm_response"        -> {action_vocale, outils}
    "alert_triggered"     -> {type, severity, details}
    "tool_executed"       -> {tool_name, result}
    "audio_transcribed"   -> {text, confidence}
    "camera_connected"    -> {source}
    "camera_disconnected" -> {source, reason}
    "camera_error"        -> {source, error}
    "entity_appeared"     -> {track_id, bbox}
    "entity_disappeared"  -> {track_id, duration}
    "entity_lingering"    -> {track_id, duration}
    "tts_speaking"        -> {text}
    "tts_done"            -> {}
    "system_status"       -> {cpu, ram, fps}
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Type alias pour les callbacks
EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class EventRecord:
    """Enregistrement d'un événement émis (pour historique/debug)."""
    event_name: str
    data: dict[str, Any]
    timestamp: float
    subscriber_count: int


class EventBus:
    """
    Bus d'événements thread-safe pour la communication inter-modules.

    Implémente le pattern Observer/Pub-Sub avec :
    - Abonnement/désabonnement dynamique.
    - Émission thread-safe (avec lock).
    - Historique optionnel des événements récents.
    - Gestion des erreurs dans les callbacks (n'interrompt pas
      les autres abonnés).

    Args:
        history_size: Nombre d'événements récents à garder en
                      historique. 0 pour désactiver.

    Example:
        >>> bus = EventBus()
        >>> results = []
        >>> bus.subscribe("test", lambda d: results.append(d))
        >>> bus.emit("test", {"key": "value"})
        >>> results
        [{'key': 'value'}]
    """

    def __init__(self, history_size: int = 100) -> None:
        self._subscribers: dict[str, list[EventCallback]] = defaultdict(
            list
        )
        self._lock = threading.RLock()
        self._history: list[EventRecord] = []
        self._history_size = history_size
        self._event_count: int = 0

    def subscribe(
        self,
        event_name: str,
        callback: EventCallback,
    ) -> None:
        """Abonne un callback à un type d'événement.

        Args:
            event_name: Nom de l'événement à écouter.
            callback: Fonction appelée quand l'événement est émis.
                      Reçoit un dict avec les données de l'événement.
        """
        with self._lock:
            if callback not in self._subscribers[event_name]:
                self._subscribers[event_name].append(callback)
                logger.debug(
                    f"Abonné à '{event_name}': "
                    f"{callback.__qualname__}"
                )

    def unsubscribe(
        self,
        event_name: str,
        callback: EventCallback,
    ) -> bool:
        """Désabonne un callback d'un type d'événement.

        Args:
            event_name: Nom de l'événement.
            callback: Le callback à retirer.

        Returns:
            True si le callback a été trouvé et retiré, False sinon.
        """
        with self._lock:
            if event_name in self._subscribers:
                try:
                    self._subscribers[event_name].remove(callback)
                    logger.debug(
                        f"Désabonné de '{event_name}': "
                        f"{callback.__qualname__}"
                    )
                    return True
                except ValueError:
                    return False
            return False

    def emit(
        self,
        event_name: str,
        data: Optional[dict[str, Any]] = None,
    ) -> int:
        """Émet un événement vers tous les abonnés.

        Les callbacks sont exécutés de manière synchrone dans le
        thread appelant. Si un callback lève une exception, elle
        est loguée mais n'interrompt pas les autres abonnés.

        Args:
            event_name: Nom de l'événement à émettre.
            data: Données associées à l'événement.

        Returns:
            Nombre de callbacks notifiés avec succès.
        """
        if data is None:
            data = {}

        with self._lock:
            callbacks = list(self._subscribers.get(event_name, []))
            self._event_count += 1

        # Exécuter les callbacks en dehors du lock
        notified = 0
        for callback in callbacks:
            try:
                callback(data)
                notified += 1
            except Exception as e:
                logger.error(
                    f"Erreur dans le callback '{callback.__qualname__}' "
                    f"pour l'événement '{event_name}': {e}",
                    exc_info=True,
                )

        # Ajouter à l'historique
        if self._history_size > 0:
            record = EventRecord(
                event_name=event_name,
                data=data,
                timestamp=time.time(),
                subscriber_count=notified,
            )
            with self._lock:
                self._history.append(record)
                if len(self._history) > self._history_size:
                    self._history = self._history[-self._history_size:]

        return notified

    def get_subscribers(self, event_name: str) -> list[EventCallback]:
        """Retourne la liste des abonnés pour un événement.

        Args:
            event_name: Nom de l'événement.

        Returns:
            Liste des callbacks abonnés (copie).
        """
        with self._lock:
            return list(self._subscribers.get(event_name, []))

    def get_all_events(self) -> list[str]:
        """Retourne la liste de tous les événements ayant
        des abonnés.

        Returns:
            Liste des noms d'événements.
        """
        with self._lock:
            return [
                name for name, subs in self._subscribers.items()
                if subs
            ]

    def get_history(
        self, event_name: Optional[str] = None, limit: int = 50
    ) -> list[EventRecord]:
        """Retourne l'historique des événements récents.

        Args:
            event_name: Si spécifié, filtre par nom d'événement.
            limit: Nombre max d'événements à retourner.

        Returns:
            Liste des EventRecord les plus récents.
        """
        with self._lock:
            history = list(self._history)

        if event_name:
            history = [r for r in history if r.event_name == event_name]

        return history[-limit:]

    @property
    def total_events(self) -> int:
        """Nombre total d'événements émis depuis le démarrage."""
        return self._event_count

    def clear(self) -> None:
        """Supprime tous les abonnés et l'historique.

        Utile pour les tests unitaires.
        """
        with self._lock:
            self._subscribers.clear()
            self._history.clear()
            self._event_count = 0

    def __repr__(self) -> str:
        with self._lock:
            sub_count = sum(
                len(subs) for subs in self._subscribers.values()
            )
        return (
            f"EventBus(events={len(self._subscribers)}, "
            f"subscribers={sub_count}, "
            f"total_emitted={self._event_count})"
        )
