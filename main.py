#!/usr/bin/env python3
"""
Sentinel-AI — Point d'entrée principal.

Gardien de sécurité autonome multimodal basé sur l'IA.
Surveille un flux vidéo, identifie les personnes, analyse le
contexte via LLM (Gemma 4), et interagit vocalement.

Usage:
    python main.py              # Démarrage normal
    python main.py --check      # Vérifier les dépendances
    python main.py --version    # Afficher la version
"""

import argparse
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

VERSION = "0.1.0"
APP_NAME = "Sentinel-AI"

ASCII_BANNER = r"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗       ║
║   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║       ║
║   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║       ║
║   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║       ║
║   ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║       ║
║   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝       ║
║               █████╗ ██╗                                  ║
║              ██╔══██╗██║                                  ║
║              ███████║██║                                  ║
║              ██╔══██║██║                                  ║
║              ██║  ██║██║                                  ║
║              ╚═╝  ╚═╝╚═╝                                  ║
║                                                           ║
║   🛡️  Gardien de Sécurité Autonome Multimodal             ║
║   📌  Version: {version:<44s}║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


# ──────────────────────────────────────────────
# Vérification des dépendances
# ──────────────────────────────────────────────

def check_dependency(
    module_name: str,
    display_name: str,
    required: bool = True,
) -> bool:
    """Vérifie si un module Python est importable.

    Args:
        module_name: Nom du module à importer.
        display_name: Nom affiché pour l'utilisateur.
        required: Si True, le module est obligatoire.

    Returns:
        True si le module est disponible, False sinon.
    """
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "?")
        status = "✅"
        print(f"  {status} {display_name:<30s} v{version}")
        return True
    except ImportError:
        status = "❌" if required else "⚠️"
        label = "MANQUANT" if required else "OPTIONNEL"
        print(f"  {status} {display_name:<30s} [{label}]")
        return False


def check_all_dependencies() -> bool:
    """Vérifie toutes les dépendances du projet.

    Returns:
        True si toutes les dépendances requises sont présentes.
    """
    print("\n🔍 Vérification des dépendances...\n")

    all_ok = True

    # ── Dépendances requises ──
    print("  ── Dépendances requises ──")
    required_deps = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("requests", "Requests"),
        ("flask", "Flask"),
    ]

    for module, name in required_deps:
        if not check_dependency(module, name, required=True):
            all_ok = False

    # ── Dépendances optionnelles ──
    print("\n  ── Dépendances optionnelles ──")
    optional_deps = [
        ("ultralytics", "Ultralytics (YOLO)"),
        ("torch", "PyTorch"),
        ("insightface", "InsightFace"),
        ("edge_tts", "Edge-TTS"),
        ("faster_whisper", "Faster-Whisper"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("fpdf2", "FPDF2"),
    ]

    for module, name in optional_deps:
        check_dependency(module, name, required=False)

    # ── Vérifier PyYAML séparément (importé comme yaml) ──
    print("\n  ── Modules internes ──")
    try:
        from src.core.config import Config
        print(f"  ✅ {'src.core.config':<30s} OK")
    except Exception as e:
        print(f"  ❌ {'src.core.config':<30s} ERREUR: {e}")
        all_ok = False

    try:
        from src.utils.logger import get_logger
        print(f"  ✅ {'src.utils.logger':<30s} OK")
    except Exception as e:
        print(f"  ❌ {'src.utils.logger':<30s} ERREUR: {e}")
        all_ok = False

    try:
        from src.utils.event_bus import EventBus
        print(f"  ✅ {'src.utils.event_bus':<30s} OK")
    except Exception as e:
        print(f"  ❌ {'src.utils.event_bus':<30s} ERREUR: {e}")
        all_ok = False

    print()
    if all_ok:
        print("  ✅ Toutes les dépendances requises sont satisfaites.")
    else:
        print("  ❌ Certaines dépendances requises sont manquantes.")
        print("     Exécutez: pip install -r requirements.txt")

    return all_ok


# ──────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande.

    Returns:
        Namespace avec les arguments parsés.
    """
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} — Gardien de Sécurité Autonome Multimodal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Vérifier les dépendances et quitter.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Afficher la version et quitter.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Chemin vers le fichier de configuration YAML.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Niveau de log (surcharge la config).",
    )
    return parser.parse_args()


def main() -> int:
    """Point d'entrée principal de Sentinel-AI.

    Returns:
        Code de sortie (0 = succès, 1 = erreur).
    """
    args = parse_args()

    # Afficher le banner
    print(ASCII_BANNER.format(version=VERSION))

    # Mode version
    if args.version:
        return 0

    # Mode vérification des dépendances
    if args.check:
        ok = check_all_dependencies()
        return 0 if ok else 1

    # ── Démarrage normal ──
    try:
        # 1. Charger la configuration
        from src.core.config import Config
        config = Config(args.config)

        # 2. Initialiser le logging
        from src.utils.logger import setup_logging, get_logger
        log_level = args.log_level or config.logging.level
        setup_logging(
            level=log_level,
            log_dir=config.logging.log_dir,
            file_enabled=config.logging.file_enabled,
            max_bytes=config.logging.max_file_size_mb * 1024 * 1024,
            backup_count=config.logging.backup_count,
        )

        logger = get_logger("main")
        logger.info(f"🛡️ {APP_NAME} v{VERSION} — Démarrage...")
        logger.info(f"📁 Racine du projet: {config.project_root}")
        logger.info(
            f"🧠 LLM: {config.llm.model_name} @ {config.llm.api_url}"
        )
        logger.info(f"📷 Caméra: source={config.camera.source}")

        # 3. Vérifier les dépendances
        logger.info("🔍 Vérification des dépendances...")
        ok = check_all_dependencies()
        if not ok:
            logger.error(
                "❌ Dépendances manquantes. "
                "Exécutez: pip install -r requirements.txt"
            )
            return 1

        # 4. Initialiser l'Event Bus
        from src.utils.event_bus import EventBus
        event_bus = EventBus()
        logger.info(f"📡 Event Bus initialisé: {event_bus}")

        # 5. Prêt !
        logger.info("✅ Sentinel-AI est prêt.")
        logger.info(
            "⏳ Les modules de perception, cognition et audio "
            "seront ajoutés dans les prochaines étapes."
        )
        logger.info(
            "💡 Utilisez --check pour vérifier les dépendances."
        )

        return 0

    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt demandé par l'utilisateur.")
        return 0

    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
