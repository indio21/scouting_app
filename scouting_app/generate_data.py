"""Generador de datos falsos para la base de datos de la aplicación de scouting.

Este script crea una base de datos SQLite y la llena con un número de
jugadores aleatorios.  Cada jugador contiene datos personales y
atributos técnicos en una escala de 0–20.  La etiqueta de potencial se
genera de manera sintética usando una lógica simple: los jugadores con
una media de habilidades por encima de cierto umbral tienen mayor
probabilidad de ser etiquetados como potenciales profesionales.

Ejemplo de uso:
    python generate_data.py --num-players 18000 --db-url sqlite:///players.db

"""

import argparse
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Player


def generate_player() -> Player:
    """Genera un jugador con atributos aleatorios."""
    names = [
        "Juan", "Pedro", "Carlos", "Mateo", "Lucas", "Gabriel",
        "Nicolás", "Diego", "Federico", "Martín", "Rodrigo", "Franco",
        "Facundo", "Santiago", "Tomás", "Julián", "Pablo", "Ignacio",
        "Marcelo", "Hernán"
    ]
    surnames = [
        "García", "López", "Martínez", "Rodríguez", "González", "Pérez",
        "Sánchez", "Romero", "Ferreira", "Suárez", "Herrera", "Ramírez",
        "Flores", "Torres", "Luna", "Álvarez", "Rojas", "Bautista",
        "Córdoba", "Vega"
    ]
    positions = ["Portero", "Defensa", "Lateral", "Mediocampista", "Delantero"]
    clubs = [
        "Club A", "Club B", "Club C", "Club D", "Club E", "Club F",
        "Academia Juvenil", "Escuela de Fútbol"
    ]
    countries = ["Argentina", "Brasil", "Uruguay", "Chile", "Paraguay"]

    name = f"{random.choice(names)} {random.choice(surnames)}"
    age = random.randint(16, 22)
    position = random.choice(positions)
    club = random.choice(clubs)
    country = random.choice(countries)

    # Generar atributos aleatorios 0–20
    attrs = {
        'pace': random.randint(0, 20),
        'shooting': random.randint(0, 20),
        'passing': random.randint(0, 20),
        'dribbling': random.randint(0, 20),
        'defending': random.randint(0, 20),
        'physical': random.randint(0, 20),
        'vision': random.randint(0, 20),
        'tackling': random.randint(0, 20),
        'determination': random.randint(0, 20),
        'technique': random.randint(0, 20),
    }

    # Calcular probabilidad de potencial profesional
    avg_skill = sum(attrs.values()) / len(attrs)
    # Umbral simple: > 12 tiene alta probabilidad
    potential = avg_skill > random.uniform(10, 15)

    return Player(
        name=name,
        age=age,
        position=position,
        club=club,
        country=country,
        pace=attrs['pace'],
        shooting=attrs['shooting'],
        passing=attrs['passing'],
        dribbling=attrs['dribbling'],
        defending=attrs['defending'],
        physical=attrs['physical'],
        vision=attrs['vision'],
        tackling=attrs['tackling'],
        determination=attrs['determination'],
        technique=attrs['technique'],
        potential_label=bool(potential),
    )


def main(num_players: int, db_url: str) -> None:
    """Genera `num_players` jugadores y los guarda en la base de datos."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    players = []
    for _ in range(num_players):
        players.append(generate_player())
    session.bulk_save_objects(players)
    session.commit()
    print(f"Se generaron {num_players} jugadores en la base de datos: {db_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera jugadores aleatorios")
    parser.add_argument("--num-players", type=int, default=1000, help="Número de jugadores a crear")
    parser.add_argument("--db-url", type=str, default="sqlite:///players.db", help="URL de la base de datos")
    args = parser.parse_args()
    main(args.num_players, args.db_url)