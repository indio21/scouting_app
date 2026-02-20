"""Utilidades compartidas para atributos y posiciones de jugadores.

Centraliza los campos, etiquetas y funciones de normalizacion usadas por
la aplicacion Flask y los scripts de entrenamiento/sincronizacion.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

ATTRIBUTE_FIELDS: List[str] = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physical",
    "vision",
    "tackling",
    "determination",
    "technique",
]

ATTRIBUTE_LABELS: Dict[str, str] = {
    "pace": "Ritmo",
    "shooting": "Disparo",
    "passing": "Pase",
    "dribbling": "Regate",
    "defending": "Defensa",
    "physical": "Fisico",
    "vision": "Vision",
    "tackling": "Marcaje",
    "determination": "Determinacion",
    "technique": "Tecnica",
}

POSITION_CHOICES: List[str] = [
    "Portero",
    "Defensa",
    "Lateral",
    "Mediocampista",
    "Delantero",
]

# Rangos de edad
EVAL_MIN_AGE = 12
EVAL_MAX_AGE = 18
MODEL_MIN_AGE = 12
MODEL_MAX_AGE = 30

# Pesos por posicion para rankear atributos especificos
POSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Portero": {
        "pace": 0.05,
        "shooting": 0.00,
        "passing": 0.10,
        "dribbling": 0.05,
        "defending": 0.20,
        "physical": 0.20,
        "vision": 0.05,
        "tackling": 0.20,
        "determination": 0.05,
        "technique": 0.10,
    },
    "Defensa": {
        "pace": 0.05,
        "shooting": 0.02,
        "passing": 0.08,
        "dribbling": 0.03,
        "defending": 0.35,
        "physical": 0.20,
        "vision": 0.05,
        "tackling": 0.15,
        "determination": 0.04,
        "technique": 0.03,
    },
    "Lateral": {
        "pace": 0.15,
        "shooting": 0.02,
        "passing": 0.10,
        "dribbling": 0.10,
        "defending": 0.25,
        "physical": 0.12,
        "vision": 0.05,
        "tackling": 0.10,
        "determination": 0.05,
        "technique": 0.06,
    },
    "Mediocampista": {
        "pace": 0.08,
        "shooting": 0.06,
        "passing": 0.25,
        "dribbling": 0.12,
        "defending": 0.12,
        "physical": 0.10,
        "vision": 0.12,
        "tackling": 0.08,
        "determination": 0.03,
        "technique": 0.04,
    },
    "Delantero": {
        "pace": 0.15,
        "shooting": 0.30,
        "passing": 0.10,
        "dribbling": 0.15,
        "defending": 0.02,
        "physical": 0.12,
        "vision": 0.06,
        "tackling": 0.00,
        "determination": 0.05,
        "technique": 0.05,
    },
}

_POSITION_KEYWORDS = (
    ("Portero", ("por", "arquero", "gk", "goalkeeper")),
    ("Defensa", ("def", "central", "cb")),
    ("Lateral", ("lat", "back", "carril", "wingback")),
    ("Mediocampista", ("med", "mid", "mix", "vol", "cm", "dmf", "amf")),
    ("Delantero", ("del", "ata", "fw", "st", "wing", "extremo")),
)


def normalized_position(value: Optional[str]) -> str:
    """Normaliza la descripcion de posicion a una etiqueta canonica."""
    if not value:
        return "Mediocampista"
    text = value.strip().lower()
    for canonical, keywords in _POSITION_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return canonical
    if text.title() in POSITION_CHOICES:
        return text.title()
    return "Mediocampista"


def position_vector(value: Optional[str]) -> List[int]:
    """One-hot de posicion para alimentar al modelo."""
    label = normalized_position(value)
    return [1 if label == pos else 0 for pos in POSITION_CHOICES]


def position_weights(value: Optional[str]) -> Dict[str, float]:
    """Devuelve el mapa de pesos segun la posicion (canonizada)."""
    label = normalized_position(value)
    return POSITION_WEIGHTS.get(label, POSITION_WEIGHTS["Mediocampista"])


def weighted_score_from_attrs(
    attrs: Dict[str, int], target_position: Optional[str] = None
) -> float:
    """Calcula un puntaje ponderado de atributos segun la posicion."""
    weights = position_weights(target_position)
    total = 0.0
    for field in ATTRIBUTE_FIELDS:
        total += float(attrs.get(field, 0) or 0) * weights[field]
    return round(total, 2)


def recommend_position_from_attrs(attrs: Dict[str, int]) -> Tuple[str, float]:
    """Devuelve la mejor posicion sugerida y su puntaje."""
    best = "Mediocampista"
    best_score = -1.0
    for pos in POSITION_CHOICES:
        score = weighted_score_from_attrs(attrs, pos)
        if score > best_score:
            best = pos
            best_score = score
    return best, round(best_score, 2)


def normalize_age_value(age: float) -> float:
    """Normaliza la edad al rango [0,1] usando los limites del modelo."""
    age_clamped = max(MODEL_MIN_AGE, min(MODEL_MAX_AGE, age))
    return (age_clamped - MODEL_MIN_AGE) / (MODEL_MAX_AGE - MODEL_MIN_AGE)


def is_valid_attribute(value: int) -> bool:
    return 0 <= value <= 20


def is_valid_eval_age(age: int) -> bool:
    return EVAL_MIN_AGE <= age <= EVAL_MAX_AGE


def _sanitize_photo_seed(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "player"


def default_player_photo_url(
    name: Optional[str] = None,
    national_id: Optional[str] = None,
    fallback: Optional[str] = None,
) -> str:
    """Retorna una URL de avatar deterministica para el jugador."""
    seed = _sanitize_photo_seed(national_id or name or fallback)
    return f"https://api.dicebear.com/9.x/adventurer/svg?seed={quote(seed)}"
