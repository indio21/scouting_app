"""Sincroniza un subconjunto de jugadores desde la base de entrenamiento."""

import argparse
from typing import List

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from models import Base, Player
from player_logic import ATTRIBUTE_FIELDS, normalized_position, default_player_photo_url


def ensure_player_columns(engine) -> None:
    with engine.connect() as conn:
        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(players)"))]
        if "national_id" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN national_id TEXT"))
        if "photo_url" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN photo_url TEXT"))


def copy_player_data(src_player: Player, dst_player: Player) -> None:
    dst_player.name = src_player.name
    dst_player.national_id = src_player.national_id
    dst_player.age = src_player.age
    dst_player.position = normalized_position(src_player.position)
    dst_player.club = src_player.club
    dst_player.country = src_player.country
    dst_player.photo_url = src_player.photo_url or default_player_photo_url(
        name=src_player.name,
        national_id=src_player.national_id,
        fallback=str(src_player.id),
    )
    dst_player.potential_label = src_player.potential_label
    for field in ATTRIBUTE_FIELDS:
        setattr(dst_player, field, getattr(src_player, field))


def sync_shortlist(src_db: str, dst_db: str, limit: int, min_age: int, max_age: int) -> None:
    src_engine = create_engine(src_db)
    dst_engine = create_engine(dst_db)
    Base.metadata.create_all(src_engine)
    Base.metadata.create_all(dst_engine)
    ensure_player_columns(src_engine)
    ensure_player_columns(dst_engine)
    SrcSession = sessionmaker(bind=src_engine)
    DstSession = sessionmaker(bind=dst_engine)

    src_session = SrcSession()
    dst_session = DstSession()
    try:
        query = (
            src_session.query(Player)
            .filter(Player.age >= min_age, Player.age <= max_age)
            .order_by(Player.potential_label.desc(), Player.age.asc(), Player.determination.desc())
        )
        players: List[Player] = query.limit(limit).all()
        synced = 0
        inserted = 0
        updated = 0
        skipped = 0
        existing_total = dst_session.query(Player).count()
        for src_player in players:
            if not src_player.national_id:
                skipped += 1
                continue
            existing = (
                dst_session.query(Player)
                .filter(Player.national_id == src_player.national_id)
                .one_or_none()
            )
            if existing:
                copy_player_data(src_player, existing)
                updated += 1
            else:
                if existing_total >= limit:
                    skipped += 1
                    continue
                new_player = Player(
                    name=src_player.name,
                    national_id=src_player.national_id,
                    age=src_player.age,
                    position=normalized_position(src_player.position),
                    club=src_player.club,
                    country=src_player.country,
                    photo_url=src_player.photo_url or default_player_photo_url(
                        name=src_player.name,
                        national_id=src_player.national_id,
                        fallback=str(src_player.id),
                    ),
                    potential_label=src_player.potential_label,
                )
                for field in ATTRIBUTE_FIELDS:
                    setattr(new_player, field, getattr(src_player, field))
                dst_session.add(new_player)
                inserted += 1
                existing_total += 1
            synced += 1
        dst_session.commit()
        print(
            f"Sincronizacion completada. actualizados={updated}, insertados={inserted}, "
            f"procesados={synced}, omitidos={skipped}, total_operativo={existing_total}, "
            f"limite={limit}, rango={min_age}-{max_age}."
        )
    finally:
        src_session.close()
        dst_session.close()


def main():
    parser = argparse.ArgumentParser(description="Copia jugadores juveniles a la base operativa.")
    parser.add_argument("--src-db", default="sqlite:///players_training.db", help="Base de datos origen (entrenamiento)")
    parser.add_argument("--dst-db", default="sqlite:///players_updated_v2.db", help="Base de datos destino (shortlist)")
    parser.add_argument("--limit", type=int, default=100, help="Cantidad maxima de jugadores a sincronizar")
    parser.add_argument("--min-age", type=int, default=12, help="Edad minima")
    parser.add_argument("--max-age", type=int, default=18, help="Edad maxima")
    args = parser.parse_args()
    if args.min_age < 10 or args.max_age < args.min_age:
        raise SystemExit("Rango de edades invalido.")
    sync_shortlist(args.src_db, args.dst_db, args.limit, args.min_age, args.max_age)


if __name__ == "__main__":
    main()
