#!/usr/bin/env python
"""Bootstrap de admin para Scouting IA (SQLite/SQLAlchemy).

Uso (desde la raíz del repo o desde scouting_app/):

  APP_DB_URL="sqlite:///players_updated_v2.db" \
  ADMIN_USERNAME="admin" \
  ADMIN_PASSWORD="admin123" \
  python scouting_app/create_admin.py

Notas:
- No pisa usuarios existentes: si el username ya existe, no hace nada y sale con código 0.
- Requiere que el esquema ya exista o que `Base.metadata.create_all()` pueda crearlo.
"""

import os
import sys

from werkzeug.security import generate_password_hash
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Importa los modelos desde scouting_app/models.py.
# Este script puede ejecutarse desde raíz o desde scouting_app/:
# agregamos el path de scouting_app al sys.path para garantizar el import.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from models import Base, User  # noqa: E402


def main() -> int:
    app_db_url = os.environ.get("APP_DB_URL", "sqlite:///players_updated_v2.db")
    admin_username = os.environ.get("ADMIN_USERNAME", "").strip()
    admin_password = os.environ.get("ADMIN_PASSWORD", "").strip()

    if not admin_username or not admin_password:
        print("ERROR: Debes setear ADMIN_USERNAME y ADMIN_PASSWORD (env vars).")
        return 2

    engine = create_engine(
        app_db_url,
        connect_args={"check_same_thread": False} if app_db_url.startswith("sqlite") else {},
    )
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        existing = db.query(User).filter(User.username == admin_username).first()
        if existing:
            print(f"OK: el usuario '{admin_username}' ya existe. No se realizaron cambios.")
            return 0

        u = User(
            username=admin_username,
            password_hash=generate_password_hash(admin_password),
            role="administrador",
        )
        db.add(u)
        db.commit()
        print(f"OK: admin creado: username='{admin_username}', role='administrador'")
        return 0
    except Exception as e:
        db.rollback()
        print("ERROR: no se pudo crear el admin:", str(e))
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
