"""
Tests unitaires pour src/utils/event_bus.py

Vérifie l'Event Bus : subscribe, emit, unsubscribe, thread-safety,
historique, et gestion d'erreurs dans les callbacks.
"""

import threading
import time
from typing import Any

import pytest

from src.utils.event_bus import EventBus, EventRecord
from src.utils.logger import reset_logging


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_logging() -> None:
    """Réinitialise le logging pour les tests."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def bus() -> EventBus:
    """Crée un EventBus frais pour chaque test."""
    return EventBus(history_size=50)


# ──────────────────────────────────────────────
# Tests — Subscribe / Emit
# ──────────────────────────────────────────────

class TestSubscribeEmit:
    """Tests d'abonnement et d'émission d'événements."""

    def test_subscribe_and_emit(self, bus: EventBus) -> None:
        """Un callback abonné reçoit les événements."""
        received: list[dict] = []
        bus.subscribe("test_event", lambda d: received.append(d))

        bus.emit("test_event", {"key": "value"})

        assert len(received) == 1
        assert received[0] == {"key": "value"}

    def test_multiple_subscribers(self, bus: EventBus) -> None:
        """Plusieurs abonnés reçoivent le même événement."""
        results_a: list[dict] = []
        results_b: list[dict] = []

        bus.subscribe("multi", lambda d: results_a.append(d))
        bus.subscribe("multi", lambda d: results_b.append(d))

        bus.emit("multi", {"data": 42})

        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_emit_without_data(self, bus: EventBus) -> None:
        """Émettre sans données envoie un dict vide."""
        received: list[dict] = []
        bus.subscribe("no_data", lambda d: received.append(d))

        bus.emit("no_data")

        assert received == [{}]

    def test_emit_returns_count(self, bus: EventBus) -> None:
        """emit() retourne le nombre de callbacks notifiés."""
        bus.subscribe("count", lambda d: None)
        bus.subscribe("count", lambda d: None)

        count = bus.emit("count", {"x": 1})
        assert count == 2

    def test_emit_no_subscribers(self, bus: EventBus) -> None:
        """Émettre un événement sans abonnés retourne 0."""
        count = bus.emit("nobody_listens", {"x": 1})
        assert count == 0

    def test_no_duplicate_subscribe(self, bus: EventBus) -> None:
        """Le même callback ne peut pas être abonné 2 fois."""
        results: list[dict] = []
        callback = lambda d: results.append(d)

        bus.subscribe("dedup", callback)
        bus.subscribe("dedup", callback)

        bus.emit("dedup", {"x": 1})
        assert len(results) == 1

    def test_different_events_isolated(self, bus: EventBus) -> None:
        """Les événements différents sont isolés."""
        results_a: list[dict] = []
        results_b: list[dict] = []

        bus.subscribe("event_a", lambda d: results_a.append(d))
        bus.subscribe("event_b", lambda d: results_b.append(d))

        bus.emit("event_a", {"from": "a"})

        assert len(results_a) == 1
        assert len(results_b) == 0


# ──────────────────────────────────────────────
# Tests — Unsubscribe
# ──────────────────────────────────────────────

class TestUnsubscribe:
    """Tests de désabonnement."""

    def test_unsubscribe(self, bus: EventBus) -> None:
        """Un callback désabonné ne reçoit plus d'événements."""
        results: list[dict] = []
        callback = lambda d: results.append(d)

        bus.subscribe("unsub", callback)
        bus.emit("unsub", {"x": 1})
        assert len(results) == 1

        bus.unsubscribe("unsub", callback)
        bus.emit("unsub", {"x": 2})
        assert len(results) == 1  # Pas de nouveau message

    def test_unsubscribe_returns_true(self, bus: EventBus) -> None:
        """unsubscribe retourne True quand le callback est trouvé."""
        callback = lambda d: None
        bus.subscribe("ret", callback)
        assert bus.unsubscribe("ret", callback) is True

    def test_unsubscribe_returns_false(self, bus: EventBus) -> None:
        """unsubscribe retourne False quand le callback n'existe pas."""
        assert bus.unsubscribe("nope", lambda d: None) is False


# ──────────────────────────────────────────────
# Tests — Gestion d'erreurs
# ──────────────────────────────────────────────

class TestErrorHandling:
    """Tests de gestion d'erreurs dans les callbacks."""

    def test_error_in_callback_doesnt_break_others(
        self, bus: EventBus
    ) -> None:
        """Une exception dans un callback n'empêche pas les autres."""
        results: list[dict] = []

        def bad_callback(data: dict) -> None:
            raise ValueError("Bug intentionnel")

        def good_callback(data: dict) -> None:
            results.append(data)

        bus.subscribe("error_test", bad_callback)
        bus.subscribe("error_test", good_callback)

        # Ne doit PAS lever d'exception
        count = bus.emit("error_test", {"x": 1})

        assert len(results) == 1  # Le bon callback a bien reçu
        assert count == 1  # Seul le bon a réussi


# ──────────────────────────────────────────────
# Tests — Thread Safety
# ──────────────────────────────────────────────

class TestThreadSafety:
    """Tests de la sécurité multi-thread."""

    def test_concurrent_emit(self, bus: EventBus) -> None:
        """Plusieurs threads peuvent émettre en parallèle."""
        counter = {"value": 0}
        lock = threading.Lock()

        def increment(data: dict) -> None:
            with lock:
                counter["value"] += 1

        bus.subscribe("concurrent", increment)

        threads = []
        for _ in range(10):
            t = threading.Thread(
                target=lambda: bus.emit("concurrent", {"x": 1})
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert counter["value"] == 10

    def test_concurrent_subscribe_emit(self, bus: EventBus) -> None:
        """Subscribe et emit peuvent être appelés en parallèle."""
        results: list[int] = []
        lock = threading.Lock()

        def record(data: dict) -> None:
            with lock:
                results.append(data.get("n", 0))

        def subscribe_and_emit(n: int) -> None:
            bus.subscribe(f"event_{n}", record)
            bus.emit(f"event_{n}", {"n": n})

        threads = [
            threading.Thread(target=subscribe_and_emit, args=(i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5


# ──────────────────────────────────────────────
# Tests — Historique
# ──────────────────────────────────────────────

class TestHistory:
    """Tests de l'historique des événements."""

    def test_history_records_events(self, bus: EventBus) -> None:
        """L'historique enregistre les événements émis."""
        bus.emit("hist_1", {"a": 1})
        bus.emit("hist_2", {"b": 2})

        history = bus.get_history()
        assert len(history) == 2
        assert history[0].event_name == "hist_1"
        assert history[1].event_name == "hist_2"

    def test_history_filter_by_name(self, bus: EventBus) -> None:
        """get_history peut filtrer par nom d'événement."""
        bus.emit("alpha", {"x": 1})
        bus.emit("beta", {"x": 2})
        bus.emit("alpha", {"x": 3})

        alpha_history = bus.get_history(event_name="alpha")
        assert len(alpha_history) == 2

    def test_history_limit(self, bus: EventBus) -> None:
        """L'historique respecte la taille max."""
        small_bus = EventBus(history_size=3)
        for i in range(5):
            small_bus.emit("test", {"i": i})

        history = small_bus.get_history()
        assert len(history) == 3

    def test_total_events(self, bus: EventBus) -> None:
        """total_events compte tous les événements émis."""
        bus.emit("a", {})
        bus.emit("b", {})
        bus.emit("c", {})
        assert bus.total_events == 3


# ──────────────────────────────────────────────
# Tests — Utilitaires
# ──────────────────────────────────────────────

class TestUtilities:
    """Tests des méthodes utilitaires."""

    def test_get_subscribers(self, bus: EventBus) -> None:
        """get_subscribers retourne la liste des callbacks."""
        cb = lambda d: None
        bus.subscribe("subs_test", cb)
        subs = bus.get_subscribers("subs_test")
        assert cb in subs

    def test_get_all_events(self, bus: EventBus) -> None:
        """get_all_events liste les événements actifs."""
        bus.subscribe("evt_1", lambda d: None)
        bus.subscribe("evt_2", lambda d: None)
        events = bus.get_all_events()
        assert "evt_1" in events
        assert "evt_2" in events

    def test_clear(self, bus: EventBus) -> None:
        """clear() vide tout."""
        bus.subscribe("clear_test", lambda d: None)
        bus.emit("clear_test", {})
        bus.clear()

        assert bus.total_events == 0
        assert bus.get_all_events() == []
        assert bus.get_history() == []

    def test_repr(self, bus: EventBus) -> None:
        """__repr__ est informatif."""
        bus.subscribe("repr_test", lambda d: None)
        repr_str = repr(bus)
        assert "EventBus(" in repr_str
        assert "subscribers=" in repr_str
