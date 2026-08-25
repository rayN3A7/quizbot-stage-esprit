"""
Migration ponctuelle : importe les comptes existants dans data/users.json
(ancien stockage) vers la base de données réelle (SQLite/PostgreSQL).

Usage :
    python scripts/migrate_users_json_to_db.py

Sans danger à relancer plusieurs fois : les utilisateurs déjà présents en
base sont simplement mis à jour (upsert), jamais dupliqués.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402
from backend.database import init_db  # noqa: E402
from backend.models import UserInDB  # noqa: E402
from backend import storage  # noqa: E402


def main():
    legacy_path = settings.DATA_DIR / "users.json"
    if not legacy_path.exists():
        print(f"Aucun fichier {legacy_path} trouvé — rien à migrer.")
        return

    init_db()
    data = json.loads(legacy_path.read_text(encoding="utf-8"))
    if not data:
        print("Le fichier users.json est vide — rien à migrer.")
        return

    migrated = 0
    for record in data.values():
        user = UserInDB(**record)
        storage.save_user(user)
        migrated += 1
        print(f"  ✓ {user.username} ({user.role.value})")

    print(f"\n{migrated} compte(s) migré(s) avec succès vers {settings.DATABASE_URL}")
    backup_path = legacy_path.with_suffix(".json.bak")
    legacy_path.rename(backup_path)
    print(f"Ancien fichier renommé en {backup_path.name} (conservé par sécurité, peut être supprimé).")


if __name__ == "__main__":
    main()
