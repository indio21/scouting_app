"""Modelos de SQLAlchemy para la aplicación de scouting.

Definimos tablas para almacenar los datos de los jugadores, sus habilidades,
estadísticas y predicciones. La estructura es sencilla para permitir una
creación rápida de prototipos; en un entorno real se recomienda
normalizar aún más los datos y añadir restricciones de integridad.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, Date, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

# Declarative base de SQLAlchemy
Base = declarative_base()


class Player(Base):
    """Tabla que representa a un jugador.

    Incluye datos personales básicos y una serie de atributos técnicos
    y mentales valorados en una escala de 0 a 20.
    """

    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    national_id = Column(String, unique=True, nullable=True)
    age = Column(Integer, nullable=False)
    position = Column(String, nullable=False)
    club = Column(String, nullable=True)
    country = Column(String, nullable=True)
    photo_url = Column(String, nullable=True)

    # Atributos técnicos y físicos (valorados 0–20)
    pace = Column(Integer, nullable=False)
    shooting = Column(Integer, nullable=False)
    passing = Column(Integer, nullable=False)
    dribbling = Column(Integer, nullable=False)
    defending = Column(Integer, nullable=False)
    physical = Column(Integer, nullable=False)
    vision = Column(Integer, nullable=False)
    tackling = Column(Integer, nullable=False)
    determination = Column(Integer, nullable=False)
    technique = Column(Integer, nullable=False)

    # Potencial real/histórico (etiqueta).  En datos generados es aleatoria.
    potential_label = Column(Boolean, nullable=False)

    stats = relationship("PlayerStat", back_populates="player", cascade="all, delete-orphan")
    attribute_history = relationship("PlayerAttributeHistory", back_populates="player", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """Convierte el modelo a un diccionario útil para JSON o plantillas."""
        return {
            "id": self.id,
            "name": self.name,
            "national_id": self.national_id,
            "age": self.age,
            "position": self.position,
            "club": self.club,
            "country": self.country,
            "photo_url": self.photo_url,
            "pace": self.pace,
            "shooting": self.shooting,
            "passing": self.passing,
            "dribbling": self.dribbling,
            "defending": self.defending,
            "physical": self.physical,
            "vision": self.vision,
            "tackling": self.tackling,
            "determination": self.determination,
            "technique": self.technique,
            "potential_label": self.potential_label,
        }

# ---------------------------------------------------------------------------
# Nuevos modelos para soporte ABM (Alta, Baja, Modificación) de cuerpos
# técnicos y dirigentes.  En un sistema real estos modelos podrían tener más
# atributos (correo electrónico, teléfono, etc.), pero para el manual de
# usuario definimos campos esenciales.

class Coach(Base):
    """Representa a un miembro del cuerpo técnico.

    Incluye información básica como nombre, rol (entrenador de porteros,
    preparador físico, etc.), edad, club y país.  El ID actúa como
    clave primaria.
    """

    __tablename__ = "coaches"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # rol dentro del cuerpo técnico
    age = Column(Integer, nullable=True)
    club = Column(String, nullable=True)
    country = Column(String, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "age": self.age,
            "club": self.club,
            "country": self.country,
        }


class Director(Base):
    """Representa a un dirigente o miembro de la directiva del club.

    A diferencia de los entrenadores, aquí usamos un campo cargo (por
    ejemplo presidente, vicepresidente, director deportivo).  También
    incluimos edad, club y país.
    """

    __tablename__ = "directors"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    position = Column(String, nullable=False)  # cargo en la directiva
    age = Column(Integer, nullable=True)
    club = Column(String, nullable=True)
    country = Column(String, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position,
            "age": self.age,
            "club": self.club,
            "country": self.country,
        }


# ---------------------------------------------------------------------------
# Modelo de usuario para autenticación

class User(Base):
    """Representa un usuario del sistema.

    Contiene un nombre de usuario único y un hash de contraseña.  En esta
    aplicación se utiliza para manejar la autenticación básica.  Para
    implementaciones más robustas se recomienda un sistema como Flask‑Login y
    almacenaje de más metadatos (roles, fechas de creación, etc.).
    """

    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="ojeador")  # roles: administrador, ojeador, director

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
        }


class PlayerStat(Base):
    """Evolucion historica de rendimiento y observaciones."""

    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    record_date = Column(Date, nullable=False)
    matches_played = Column(Integer, nullable=False, default=0)
    goals = Column(Integer, nullable=False, default=0)
    assists = Column(Integer, nullable=False, default=0)
    minutes_played = Column(Integer, nullable=False, default=0)
    yellow_cards = Column(Integer, nullable=False, default=0)
    red_cards = Column(Integer, nullable=False, default=0)
    pass_accuracy = Column(Float, nullable=True)  # porcentaje (0-100)
    shot_accuracy = Column(Float, nullable=True)  # porcentaje (0-100)
    duels_won_pct = Column(Float, nullable=True)  # porcentaje (0-100)
    final_score = Column(Float, nullable=True)    # valoración 1-10
    notes = Column(Text, nullable=True)

    player = relationship("Player", back_populates="stats")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "player_id": self.player_id,
            "record_date": self.record_date.isoformat(),
            "matches_played": self.matches_played,
            "goals": self.goals,
            "assists": self.assists,
            "minutes_played": self.minutes_played,
            "yellow_cards": self.yellow_cards,
            "red_cards": self.red_cards,
            "pass_accuracy": self.pass_accuracy,
            "shot_accuracy": self.shot_accuracy,
            "duels_won_pct": self.duels_won_pct,
            "final_score": self.final_score,
            "notes": self.notes,
        }


class PlayerAttributeHistory(Base):
    """Historial de atributos técnicos/mentales para seguir progresión."""

    __tablename__ = "player_attribute_history"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    record_date = Column(Date, nullable=False)
    pace = Column(Integer, nullable=True)
    shooting = Column(Integer, nullable=True)
    passing = Column(Integer, nullable=True)
    dribbling = Column(Integer, nullable=True)
    defending = Column(Integer, nullable=True)
    physical = Column(Integer, nullable=True)
    vision = Column(Integer, nullable=True)
    tackling = Column(Integer, nullable=True)
    determination = Column(Integer, nullable=True)
    technique = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)

    player = relationship("Player", back_populates="attribute_history")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "player_id": self.player_id,
            "record_date": self.record_date.isoformat(),
            "pace": self.pace,
            "shooting": self.shooting,
            "passing": self.passing,
            "dribbling": self.dribbling,
            "defending": self.defending,
            "physical": self.physical,
            "vision": self.vision,
            "tackling": self.tackling,
            "determination": self.determination,
            "technique": self.technique,
            "notes": self.notes,
        }
