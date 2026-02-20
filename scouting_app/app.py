import os
import logging
import time
import threading
import secrets
import sys
import subprocess
from typing import List, Tuple, Callable, Optional, Dict, Set
from datetime import datetime, date, timedelta
from statistics import mean
from types import SimpleNamespace
from flask import Flask, render_template, redirect, url_for, request, session, flash, abort, jsonify
from sqlalchemy import create_engine, func, desc, text
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker, joinedload
import numpy as np
import torch
from models import Base, Player, Coach, Director, User, PlayerStat, PlayerAttributeHistory
from train_model import PlayerNet, normalize_features
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from player_logic import (
    ATTRIBUTE_FIELDS,
    ATTRIBUTE_LABELS,
    POSITION_CHOICES,
    normalized_position,
    position_vector,
    position_weights,
    recommend_position_from_attrs,
    weighted_score_from_attrs,
    is_valid_attribute,
    is_valid_eval_age,
    default_player_photo_url,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Límite de payload para requests (mitiga abusos y errores por uploads grandes)
# Default: 2MB. Ajustable por env var `MAX_CONTENT_LENGTH`.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", str(2 * 1024 * 1024)))


# --- Observabilidad mínima ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
try:
    app.logger.setLevel(LOG_LEVEL)
except Exception:
    app.logger.setLevel(logging.INFO)

# Logger root para librerías (opcional)
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))


# --- Cache in-memory con TTL (MVP) ---
_CACHE: dict = {}
_CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "60"))


# --- Guardrails pipeline (evita doble ejecución concurrente) ---
_PIPELINE_LOCK = threading.Lock()


def _cache_get(key: str):
    item = _CACHE.get(key)
    if not item:
        return None
    expires_at, value = item
    if time.time() >= expires_at:
        _CACHE.pop(key, None)
        return None
    return value

def _cache_set(key: str, value):
    _CACHE[key] = (time.time() + _CACHE_TTL_SECONDS, value)


app.secret_key = os.environ.get("APP_SECRET_KEY", "reemplazar-esta-clave")


# --- Secret key obligatoria en producción ---
_env2 = (os.environ.get("FLASK_ENV") or os.environ.get("ENV") or "").lower()
_secret = os.environ.get("APP_SECRET_KEY", "")
if _env2 in ("production", "prod") and (not _secret or _secret == "reemplazar-esta-clave"):
    raise RuntimeError("APP_SECRET_KEY must be set in production")


# --- Cookies de sesión (hardening mínimo) ---
_env = (os.environ.get("FLASK_ENV") or os.environ.get("ENV") or "").lower()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
if _env in ("production", "prod"):
    app.config["SESSION_COOKIE_SECURE"] = True


# --- CSRF mínimo (session token) ---
def _csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token

@app.context_processor
def inject_csrf_token():
    return {"csrf_token": _csrf_token()}

def _require_csrf():
    # Solo para endpoints críticos (llamar manualmente en POST)
    token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if not token or token != session.get("csrf_token"):
        abort(400)

def _resolve_sqlite_url(url: str) -> str:
    # sqlite:///file.db  -> relativo (se resuelve dentro de scouting_app/)
    # sqlite:////abs/path/file.db -> absoluto (no se toca)
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = url.replace("sqlite:///", "", 1)
        return "sqlite:///" + os.path.join(BASE_DIR, rel).replace("\\", "/")
    return url

APP_DB_URL = _resolve_sqlite_url(os.environ.get("APP_DB_URL", "sqlite:///players_updated_v2.db"))
if APP_DB_URL.rsplit("/", 1)[-1] == "players.db":
    app.logger.warning("APP_DB_URL apunta a players.db (legacy). Se recomienda players_updated_v2.db. Evidencia: scouting_app/players.db")

TRAINING_DB_URL = _resolve_sqlite_url(os.environ.get("TRAINING_DB_URL", "sqlite:///players_training.db"))
try:
    EVAL_POOL_MAX = max(1, int(os.environ.get("EVAL_POOL_MAX", "100")))
except ValueError:
    EVAL_POOL_MAX = 100

SYNC_SHORTLIST_ENABLED = (os.environ.get("SYNC_SHORTLIST_ENABLED", "0").strip().lower() in {
    "1", "true", "yes", "y", "si", "s", "on"
})

ENFORCE_EVAL_POOL_LIMIT = (os.environ.get("ENFORCE_EVAL_POOL_LIMIT", "1").strip().lower() in {
    "1", "true", "yes", "y", "si", "s", "on"
})

engine = create_engine(APP_DB_URL, connect_args={"check_same_thread": False} if APP_DB_URL.startswith("sqlite") else {})
if APP_DB_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.close()

Session = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(engine)


def ensure_player_schema():
    """Agrega columnas nuevas si la base ya existia."""
    with engine.connect() as conn:
        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(players)"))]
        if "national_id" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN national_id TEXT"))
        if "photo_url" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN photo_url TEXT"))


ensure_player_schema()


def trim_operational_player_pool(db_session, max_players: int = EVAL_POOL_MAX) -> int:
    """Mantiene la base operativa en un maximo de jugadores evaluables."""
    total = db_session.query(func.count(Player.id)).scalar() or 0
    if total <= max_players:
        return 0

    stat_subq = (
        db_session.query(
            PlayerStat.player_id.label("player_id"),
            func.max(PlayerStat.record_date).label("last_stat_date"),
        )
        .group_by(PlayerStat.player_id)
        .subquery()
    )
    attr_subq = (
        db_session.query(
            PlayerAttributeHistory.player_id.label("player_id"),
            func.max(PlayerAttributeHistory.record_date).label("last_attr_date"),
        )
        .group_by(PlayerAttributeHistory.player_id)
        .subquery()
    )

    rows = (
        db_session.query(
            Player.id,
            Player.age,
            stat_subq.c.last_stat_date,
            attr_subq.c.last_attr_date,
        )
        .outerjoin(stat_subq, stat_subq.c.player_id == Player.id)
        .outerjoin(attr_subq, attr_subq.c.player_id == Player.id)
        .all()
    )

    def sort_key(row):
        dates = [d for d in (row.last_stat_date, row.last_attr_date) if d is not None]
        last_activity = max(dates) if dates else date.min
        has_history = 1 if dates else 0
        in_eval_range = 1 if is_valid_eval_age(int(row.age or 0)) else 0
        return (has_history, in_eval_range, last_activity, row.id)

    rows_sorted = sorted(rows, key=sort_key, reverse=True)
    keep_ids: Set[int] = {row.id for row in rows_sorted[:max_players]}
    drop_ids = [row.id for row in rows_sorted if row.id not in keep_ids]

    if not drop_ids:
        return 0

    db_session.query(PlayerStat).filter(PlayerStat.player_id.in_(drop_ids)).delete(synchronize_session=False)
    db_session.query(PlayerAttributeHistory).filter(PlayerAttributeHistory.player_id.in_(drop_ids)).delete(synchronize_session=False)
    db_session.query(Player).filter(Player.id.in_(drop_ids)).delete(synchronize_session=False)
    return len(drop_ids)


def backfill_player_photo_urls(db_session) -> int:
    players = (
        db_session.query(Player)
        .filter((Player.photo_url == None) | (Player.photo_url == ""))  # noqa: E711
        .all()
    )
    updated = 0
    for player in players:
        player.photo_url = default_player_photo_url(
            name=player.name,
            national_id=player.national_id,
            fallback=str(player.id),
        )
        updated += 1
    return updated


def enforce_operational_pool_limit_on_startup():
    if not ENFORCE_EVAL_POOL_LIMIT:
        return
    db = Session()
    try:
        removed = trim_operational_player_pool(db, EVAL_POOL_MAX)
        if removed:
            db.commit()
            app.logger.warning(
                "Base operativa recortada a %s jugadores (eliminados: %s).",
                EVAL_POOL_MAX,
                removed,
            )
        photo_updates = backfill_player_photo_urls(db)
        if photo_updates:
            db.commit()
            app.logger.info("Fotos de jugadores completadas: %s", photo_updates)
        history_sync_fn = globals().get("sync_attribute_history_baseline")
        if callable(history_sync_fn):
            history_updates = history_sync_fn(db)
            if history_updates:
                db.commit()
                app.logger.info("Historial tecnico sincronizado desde ficha actual: %s jugadores", history_updates)
    except Exception:
        db.rollback()
        app.logger.exception("No se pudo aplicar el limite de jugadores operativos al iniciar.")
    finally:
        db.close()


def run_subprocess(command: List[str], description: str) -> Tuple[bool, str]:
    """Ejecuta un comando externo y devuelve (exito, mensaje)."""
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        log = f"[{description}] {'OK' if result.returncode == 0 else 'ERROR'}"
        if output:
            log += f"\n{output}"
        if error:
            log += f"\n{error}"
        return result.returncode == 0, log
    except Exception as exc:
        return False, f"[{description}] ERROR: {exc}"


def ensure_training_dataset(min_players: int = 1) -> Tuple[bool, List[str]]:
    """Verifica que la BD de entrenamiento tenga datos; genera si esta vacia."""
    logs: List[str] = []
    training_engine = create_engine(TRAINING_DB_URL)
    TrainingSession = sessionmaker(bind=training_engine)
    Base.metadata.create_all(training_engine)
    session = TrainingSession()
    try:
        count = session.query(func.count(Player.id)).scalar() or 0
    finally:
        session.close()
    logs.append(f"Jugadores disponibles en base de referencia: {count}")
    if count >= min_players:
        return True, logs
    cmd = [
        sys.executable,
        "generate_data.py",
        "--num-players",
        "20000",
        "--db-url",
        TRAINING_DB_URL,
    ]
    success, log = run_subprocess(cmd, "Preparar base de referencia")
    logs.append(log)
    return success, logs


def update_database_pipeline(limit: int = EVAL_POOL_MAX, sync_shortlist: bool = SYNC_SHORTLIST_ENABLED) -> Tuple[bool, List[str]]:
    """Ejecuta entrenamiento del modelo y sincronizacion opcional para la base operativa."""
    overall_logs: List[str] = []
    ok, logs = ensure_training_dataset()
    overall_logs.extend(logs)
    if not ok:
        return False, overall_logs
    train_cmd = [
        sys.executable,
        "train_model.py",
        "--db-url",
        TRAINING_DB_URL,
        "--epochs",
        "30",
    ]
    ok, train_log = run_subprocess(train_cmd, "Actualizacion de puntajes")
    overall_logs.append(train_log)
    if not ok:
        return False, overall_logs

    if sync_shortlist:
        ensure_player_schema()
        sync_cmd = [
            sys.executable,
            "sync_shortlist.py",
            "--src-db",
            TRAINING_DB_URL,
            "--dst-db",
            APP_DB_URL,
            "--limit",
            str(limit),
        ]
        ok, sync_log = run_subprocess(sync_cmd, "Sincronizacion de base operativa")
        overall_logs.append(sync_log)
        if not ok:
            return False, overall_logs
    else:
        overall_logs.append("Sincronizacion de base operativa omitida.")

    # Guardrail final: la base operativa no debe superar EVAL_POOL_MAX.
    db = Session()
    try:
        removed = trim_operational_player_pool(db, EVAL_POOL_MAX)
        if removed:
            db.commit()
            overall_logs.append(f"Base operativa recortada a {EVAL_POOL_MAX} jugadores (eliminados {removed}).")
        else:
            db.rollback()
            overall_logs.append(f"Base operativa dentro del limite ({EVAL_POOL_MAX} jugadores).")
    except Exception as exc:
        db.rollback()
        overall_logs.append(f"No se pudo aplicar el recorte de base operativa: {exc}")
        return False, overall_logs
    finally:
        db.close()

    return True, overall_logs

# ----------------------------------------------------
# Usuario administrador inicial (opcional en desarrollo)
def init_admin_user():
    username = (os.environ.get("ADMIN_USERNAME") or "admin").strip()
    password = (os.environ.get("ADMIN_PASSWORD") or "").strip()
    allow_default = (os.environ.get("ALLOW_DEFAULT_ADMIN") or "").strip().lower() in {
        "1", "true", "yes", "y", "si", "s", "on"
    }
    is_prod = (os.environ.get("FLASK_ENV") or os.environ.get("ENV") or "").lower() in {"production", "prod"}

    if not password:
        # En produccion no crear admin por defecto.
        if is_prod:
            app.logger.warning("ADMIN_PASSWORD no configurado. No se crea usuario admin inicial en produccion.")
            return
        # En desarrollo, solo permitir admin por defecto si se habilita explicito.
        if not allow_default:
            app.logger.info(
                "ADMIN_PASSWORD no configurado. Se omite bootstrap de admin. "
                "Setear ADMIN_PASSWORD o ALLOW_DEFAULT_ADMIN=true para desarrollo local."
            )
            return
        password = "admin"

    if len(username) < 3 or len(password) < 6:
        app.logger.warning("Credenciales de bootstrap invalidas. No se crea usuario admin inicial.")
        return

    db = Session()
    existing = db.query(User).filter(User.username == username).first()
    if not existing:
        user = User(username=username,
                    password_hash=generate_password_hash(password),
                    role="administrador")
        db.add(user)
        db.commit()
    db.close()

init_admin_user()
enforce_operational_pool_limit_on_startup()

# ----------------------------------------------------
# Landing
@app.route("/")
def landing():
    db = Session()
    total_players = db.query(func.count(Player.id)).scalar() or 0
    avg_age = db.query(func.avg(Player.age)).scalar()
    countries = db.query(Player.country).distinct().count()
    positions = db.query(Player.position).distinct().count()
    db.close()

    metrics = {
        "total_players": int(total_players),
        "avg_age": float(avg_age) if avg_age else None,
        "countries": countries,
        "positions": positions,
    }

    if session.get("user_id"):
        call_to_action_url = url_for("index")
        call_to_action_label = "Ir al panel"
    else:
        call_to_action_url = url_for("login")
        call_to_action_label = "Iniciar sesión"

    return render_template(
        "landing.html",
        metrics=metrics,
        call_to_action_url=call_to_action_url,
        call_to_action_label=call_to_action_label,
    )

# ----------------------------------------------------

@app.context_processor
def navbar_url_helpers():
    def first_url(*endpoints, **values):
        """Devuelve la primera URL válida entre una lista de endpoints.
        Si ninguno existe, devuelve '#'.
        """
        for ep in endpoints:
            try:
                return url_for(ep, **values)
            except Exception:
                continue
        return "#"
    return dict(first_url=first_url)

@app.context_processor
def auth_flags():
    return {
        "is_authenticated": bool(session.get("user_id")),
        "current_username": session.get("username", "admin"),
    }


def display_position_label(value: Optional[str]) -> str:
    normalized = normalized_position(value)
    return "Arquero" if normalized == "Portero" else normalized


@app.context_processor
def position_labels():
    return {"display_position": display_position_label}

# Decorador de login
def login_required(view_func: Callable) -> Callable:
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login', next=request.url))
        return view_func(*args, **kwargs)
    return wrapper

# ----------------------------------------------------
# Modelo
def load_model(model_path: str) -> PlayerNet:
    input_dim = 1 + len(ATTRIBUTE_FIELDS) + len(POSITION_CHOICES)
    model = PlayerNet(input_dim=input_dim)
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        print("Advertencia: modelo incompatible con la arquitectura actual. Intentando reentrenar automaticamente.")
        print(exc)
        success, logs = update_database_pipeline()
        for log in logs:
            print(log)
        if not success:
            raise RuntimeError("No se pudo reentrenar el modelo automaticamente. Ejecute train_model.py manualmente.") from exc
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        except Exception as exc2:
            raise RuntimeError("No se pudo cargar el modelo incluso despues de reentrenar.") from exc2
    model.eval()
    return model
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    model = None
    print("Advertencia: modelo no encontrado.")

@app.route("/health")
def health():
    """Healthcheck básico: app viva + conectividad DB."""
    try:
        # Validación mínima de conectividad
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        app.logger.exception("Healthcheck failed")
        return jsonify({"status": "error", "detail": str(e)}), 500



def prepare_input(player: Player) -> torch.Tensor:
    features = [
        player.age, player.pace, player.shooting, player.passing, player.dribbling,
        player.defending, player.physical, player.vision, player.tackling,
        player.determination, player.technique
    ]
    features.extend(position_vector(player.position))
    X = normalize_features(torch.tensor([features], dtype=torch.float32).numpy())
    return torch.tensor(X, dtype=torch.float32)

def compute_suggestions(player: Player, threshold=14, top_n=3) -> List[Tuple[str, int]]:
    attrs = {
        "Ritmo (pace)": player.pace,
        "Disparo (shooting)": player.shooting,
        "Pase (passing)": player.passing,
        "Regate (dribbling)": player.dribbling,
        "Defensa (defending)": player.defending,
        "Físico (physical)": player.physical,
        "Visión (vision)": player.vision,
        "Marcaje (tackling)": player.tackling,
        "Determinación (determination)": player.determination,
        "Técnica (technique)": player.technique,
    }
    gaps = {name: threshold - value for name, value in attrs.items() if value < threshold}
    return sorted(gaps.items(), key=lambda item: item[1], reverse=True)[:top_n]


def score_band(score: float) -> str:
    if score >= 15:
        return "Alto"
    if score >= 10:
        return "Medio"
    return "Bajo"

POSITION_OPTIONS = POSITION_CHOICES


def player_attribute_map(player: Player) -> Dict[str, int]:
    return {field: getattr(player, field) for field in ATTRIBUTE_FIELDS}


def player_feature_vector(player: Player) -> List[float]:
    values = [player.age]
    values.extend(getattr(player, field) for field in ATTRIBUTE_FIELDS)
    values.extend(position_vector(player.position))
    return values


def player_fit_score(player: Player, target_position: Optional[str] = None) -> float:
    return weighted_score_from_attrs(player_attribute_map(player), target_position)


def normalize_identifier(raw_value: Optional[str]) -> Optional[str]:
    if not raw_value:
        return None
    digits = "".join(ch for ch in raw_value if ch.isdigit())
    if len(digits) < 6:
        return None
    return digits



def fetch_player_stats(player_id: int, db_session = None) -> List[PlayerStat]:
    close_session = False
    if db_session is None:
        db_session = Session()
        close_session = True
    stats = (db_session.query(PlayerStat)
             .filter(PlayerStat.player_id == player_id)
             .order_by(PlayerStat.record_date.asc(), PlayerStat.id.asc())
             .all())
    if close_session:
        db_session.close()
    return stats


def summarize_stats(stats: List[PlayerStat]) -> Dict[str, Optional[float]]:
    if not stats:
        return {
            "entries": 0,
            "total_matches": 0,
            "total_goals": 0,
            "total_assists": 0,
            "total_minutes": 0,
            "avg_pass_accuracy": None,
            "avg_shot_accuracy": None,
            "avg_duels": None,
            "avg_final_score": None,
            "latest_date": None,
        }

    def avg(values: List[Optional[float]]) -> Optional[float]:
        filtered = [v for v in values if v is not None]
        return round(mean(filtered), 2) if filtered else None

    return {
        "entries": len(stats),
        "total_matches": sum(s.matches_played for s in stats),
        "total_goals": sum(s.goals for s in stats),
        "total_assists": sum(s.assists for s in stats),
        "total_minutes": sum(s.minutes_played for s in stats),
        "avg_pass_accuracy": avg([s.pass_accuracy for s in stats]),
        "avg_shot_accuracy": avg([s.shot_accuracy for s in stats]),
        "avg_duels": avg([s.duels_won_pct for s in stats]),
        "avg_final_score": avg([s.final_score for s in stats if s.final_score is not None]),
        "latest_date": stats[-1].record_date.isoformat(),
    }


def fetch_attribute_history(player_id: int, db_session = None) -> List[PlayerAttributeHistory]:
    close_session = False
    if db_session is None:
        db_session = Session()
        close_session = True
    history = (db_session.query(PlayerAttributeHistory)
               .filter(PlayerAttributeHistory.player_id == player_id)
               .order_by(PlayerAttributeHistory.record_date.asc(), PlayerAttributeHistory.id.asc())
               .all())
    if close_session:
        db_session.close()
    return history


def _attribute_row_from_player(player: Player) -> Dict[str, int]:
    return {field: int(getattr(player, field) or 0) for field in ATTRIBUTE_FIELDS}


def _attribute_row_from_entry(entry: PlayerAttributeHistory) -> Dict[str, int]:
    return {field: int(getattr(entry, field) or 0) for field in ATTRIBUTE_FIELDS}


def sync_player_attribute_history(player: Player, db_session, note: str = "Sincronizacion automatica de ficha") -> bool:
    """Asegura que el ultimo registro del historial tecnico refleje la ficha actual.

    Devuelve True si crea un registro nuevo.
    """
    latest = (
        db_session.query(PlayerAttributeHistory)
        .filter(PlayerAttributeHistory.player_id == player.id)
        .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
        .first()
    )
    current_values = _attribute_row_from_player(player)
    if latest and _attribute_row_from_entry(latest) == current_values:
        return False

    entry = PlayerAttributeHistory(
        player_id=player.id,
        record_date=date.today(),
        notes=note,
        **current_values,
    )
    db_session.add(entry)
    return True


def sync_attribute_history_baseline(db_session) -> int:
    """Sincroniza historial tecnico para jugadores con ficha desfasada o sin historial."""
    players = db_session.query(Player).all()
    created = 0
    for player in players:
        if sync_player_attribute_history(player, db_session):
            created += 1
    return created


def summarize_attribute_history(history: List[PlayerAttributeHistory]) -> Dict[str, Optional[int]]:
    if not history:
        summary: Dict[str, Optional[int]] = {field: None for field in ATTRIBUTE_FIELDS}
        summary["entries"] = 0
        summary["latest_date"] = None
        return summary
    latest = history[-1]
    summary = {field: getattr(latest, field) for field in ATTRIBUTE_FIELDS}
    summary["entries"] = len(history)
    summary["latest_date"] = latest.record_date.isoformat()
    return summary


def combine_probability(base_prob: float, stats_summary: Dict[str, Optional[float]], fit_score: Optional[float] = None) -> float:
    """Combina la probabilidad del modelo con señales simples del historial y del fit del jugador.

    - base_prob: salida del modelo (0..1)
    - avg_final_score: promedio histórico (1..10) si existe
    - fit_score: puntaje ponderado por posición (0..20) si existe
    """
    avg_score = stats_summary.get("avg_final_score")
    rating_weight = None if avg_score is None else min(max(float(avg_score) / 10.0, 0.0), 1.0)

    fit_weight = None
    if fit_score is not None:
        try:
            fit_weight = min(max(float(fit_score) / 20.0, 0.0), 1.0)
        except Exception:
            fit_weight = None

    # Pesos (tuneables vía env vars)
    w_model = float(os.environ.get("POT_W_MODEL", "0.35"))
    w_rating = float(os.environ.get("POT_W_RATING", "0.35"))
    w_fit = float(os.environ.get("POT_W_FIT", "0.30"))

    # Si no hay rating o fit, re-normalizamos para no castigar por falta de datos
    components = [(w_model, base_prob)]
    if rating_weight is not None:
        components.append((w_rating, rating_weight))
    if fit_weight is not None:
        components.append((w_fit, fit_weight))

    weight_sum = sum(w for w, _ in components) or 1.0
    combined = sum(w * v for w, v in components) / weight_sum

    return max(0.0, min(combined, 0.99))
def stats_chart_payload(stats: List[PlayerStat]) -> Dict[str, List]:
    labels = []
    final_scores = []
    pass_pct = []
    shot_pct = []
    duel_pct = []
    for entry in stats:
        labels.append(entry.record_date.strftime("%Y-%m-%d"))
        final_scores.append(entry.final_score if entry.final_score is not None else None)
        pass_pct.append(entry.pass_accuracy if entry.pass_accuracy is not None else None)
        shot_pct.append(entry.shot_accuracy if entry.shot_accuracy is not None else None)
        duel_pct.append(entry.duels_won_pct if entry.duels_won_pct is not None else None)
    return {
        "labels": labels,
        "final_scores": final_scores,
        "pass_pct": pass_pct,
        "shot_pct": shot_pct,
        "duel_pct": duel_pct,
    }


def calculate_stats_rating(metrics: Dict[str, Optional[float]]) -> float:
    matches = metrics.get("matches", 0) or 0
    goals = metrics.get("goals", 0) or 0
    assists = metrics.get("assists", 0) or 0
    minutes = metrics.get("minutes", 0) or 0
    pass_pct = metrics.get("pass_pct") or 0.0
    shot_pct = metrics.get("shot_pct") or 0.0
    duels_pct = metrics.get("duels_pct") or 0.0

    minutes_factor = min(minutes / 90.0, 1.5)
    scoring_factor = min(goals, 3) * 1.5 + min(assists, 3) * 1.2
    accuracy_factor = (pass_pct / 100.0) * 2.0 + (shot_pct / 100.0) * 1.5 + (duels_pct / 100.0) * 1.3
    consistency_factor = min(matches, 3) * 0.5

    raw_score = 1.5 + minutes_factor + scoring_factor + accuracy_factor + consistency_factor
    return round(max(1.0, min(10.0, raw_score)), 2)


def attribute_chart_payload(history: List[PlayerAttributeHistory]) -> Dict[str, List[Optional[int]]]:
    labels: List[str] = []
    series: Dict[str, List[Optional[int]]] = {field: [] for field in ATTRIBUTE_FIELDS}
    for entry in history:
        labels.append(entry.record_date.strftime("%Y-%m-%d"))
        for field in ATTRIBUTE_FIELDS:
            series[field].append(getattr(entry, field))
    return {"labels": labels, "series": series}


def categorize_probability(probability: float) -> str:
    high = float(os.environ.get("POTENTIAL_HIGH_THRESHOLD", "0.60"))
    medium = float(os.environ.get("POTENTIAL_MEDIUM_THRESHOLD", "0.35"))
    if medium >= high:
        medium = max(0.0, high - 0.05)

    if probability >= high:
        return "Alto potencial"
    if probability >= medium:
        return "Potencial medio"
    return "Bajo potencial"
def compute_projection(player: Player, stats: Optional[List[PlayerStat]] = None, db_session = None) -> Optional[Dict[str, object]]:
    if model is None:
        return None
    if stats is None:
        stats_list = fetch_player_stats(player.id, db_session=db_session)
    else:
        stats_list = stats
    stats_summary = summarize_stats(stats_list)
    input_tensor = prepare_input(player)
    with torch.no_grad():
        base_prob = model(input_tensor).item()
    attr_map = player_attribute_map(player)
    best_position, best_score = recommend_position_from_attrs(attr_map)
    fit_score = weighted_score_from_attrs(attr_map, player.position)
    combined = combine_probability(base_prob, stats_summary, fit_score=fit_score)
    return {
        "base_prob": base_prob,
        "combined_prob": combined,
        "category": categorize_probability(combined),
        "stats_summary": stats_summary,
        "history": stats_list,
        "fit_score": fit_score,
        "recommended_position": best_position,
        "recommended_score": best_score,
    }


def refresh_player_potential(player: Player, db_session = None) -> Optional[Dict[str, object]]:
    projection = compute_projection(player, db_session=db_session)
    if projection:
        player.potential_label = projection["combined_prob"] >= 0.7
    return projection

# ----------------------------------------------------
# LOGIN
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        _require_csrf()

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        db = Session()
        user = db.query(User).filter(User.username == username).first()
        db.close()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            return redirect(request.args.get('next') or url_for('index'))
        return render_template('login.html', error='Usuario o contraseña inválidos')
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))  # <- antes apuntaba al login; ahora a la web pública


# ----------------------------------------------------
# REGISTRO
@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == "POST":
        _require_csrf()

    if session.get('role') != 'administrador':
        return "No autorizado", 403
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')
        if not username or not password or not role:
            return render_template('register.html', error='Todos los campos son obligatorios')
        db = Session()
        if db.query(User).filter(User.username == username).first():
            db.close()
            return render_template('register.html', error='El usuario ya existe')
        user = User(username=username,
                    password_hash=generate_password_hash(password),
                    role=role)
        db.add(user)
        db.commit()
        db.close()
        return redirect(url_for('index'))
    return render_template('register.html')

# ----------------------------------------------------
# DASHBOARD
@app.route('/dashboard')
@login_required
def dashboard():
    period = request.args.get('period', 'month')
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    dashboard_cache_key = f"dashboard:{period}:{start_str}:{end_str}"
    cached_html = _cache_get(dashboard_cache_key)
    if cached_html is not None:
        return cached_html

    today = date.today()
    if period == 'custom':
        try:
            start_date = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else today - timedelta(days=30)
        except ValueError:
            start_date = today - timedelta(days=30)
        try:
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else today
        except ValueError:
            end_date = today
    else:
        end_date = today
        delta = {
            'week': 7,
            'month': 30,
            'year': 365
        }.get(period, 30)
        start_date = end_date - timedelta(days=delta)
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    db = Session()
    pos_rows = db.query(Player.position, func.count(Player.id)).group_by(Player.position).all()
    positions = [display_position_label(row[0]) if row[0] else "Sin posicion" for row in pos_rows]
    pos_values = [row[1] for row in pos_rows]
    total_players = db.query(func.count(Player.id)).scalar() or 0
    avg_attrs = db.query(
        func.avg(Player.pace), func.avg(Player.shooting), func.avg(Player.passing),
        func.avg(Player.dribbling), func.avg(Player.defending), func.avg(Player.physical),
        func.avg(Player.vision), func.avg(Player.tackling), func.avg(Player.determination), func.avg(Player.technique)
    ).one()
    avg_labels = [ATTRIBUTE_LABELS[field] for field in ATTRIBUTE_FIELDS]
    avg_values = [round(float(value), 2) if value is not None else 0 for value in avg_attrs]

    players = db.query(Player).all()
    player_avg_rows = db.query(
        PlayerStat.player_id, func.avg(PlayerStat.final_score)
    ).group_by(PlayerStat.player_id).all()
    avg_score_map = {
        player_id: (float(avg_score) if avg_score is not None else None)
        for player_id, avg_score in player_avg_rows
    }
    top_potential = []
    if players and model is not None:
        features = np.array([
            player_feature_vector(player)
            for player in players
        ], dtype=np.float32)
        features_norm = normalize_features(features)
        with torch.no_grad():
            probs_tensor = model(torch.tensor(features_norm))
        base_probs = probs_tensor.squeeze().numpy().tolist()
        for idx, player in enumerate(players):
            base_prob = float(base_probs[idx])
            stats_summary = {"avg_final_score": avg_score_map.get(player.id)}
            attr_map = player_attribute_map(player)
            fit_score = weighted_score_from_attrs(attr_map, player.position)
            combined = combine_probability(base_prob, stats_summary, fit_score=fit_score)
            top_potential.append({
                "id": player.id,
                "name": player.name,
                "probability": combined,
                "category": categorize_probability(combined),
            })
        top_potential.sort(key=lambda item: item["probability"], reverse=True)
        top_potential = top_potential[:10]

    stats_in_range = (db.query(PlayerStat)
                      .filter(PlayerStat.record_date >= start_date,
                              PlayerStat.record_date <= end_date,
                              PlayerStat.final_score != None)
                      .order_by(PlayerStat.player_id.asc(),
                                PlayerStat.record_date.asc(),
                                PlayerStat.id.asc())
                      .all())
    player_map = {player.id: player for player in players}
    evolution_map: Dict[int, Dict[str, object]] = {}
    for stat in stats_in_range:
        entry = evolution_map.setdefault(
            stat.player_id,
            {
                "first_score": stat.final_score,
                "first_date": stat.record_date,
                "last_score": stat.final_score,
                "last_date": stat.record_date,
            }
        )
        if stat.record_date < entry["first_date"]:
            entry["first_date"] = stat.record_date
            entry["first_score"] = stat.final_score
        if stat.record_date >= entry["last_date"]:
            entry["last_date"] = stat.record_date
            entry["last_score"] = stat.final_score

    top_evolution = []
    for player_id, values in evolution_map.items():
        first = values["first_score"]
        last = values["last_score"]
        if first is None or last is None:
            continue
        delta = round(float(last - first), 2)
        top_evolution.append({
            "id": player_id,
            "name": player_map[player_id].name if player_id in player_map else f"Jugador {player_id}",
            "delta": delta,
            "start": values["first_date"],
            "end": values["last_date"],
        })
    top_evolution.sort(key=lambda item: item["delta"], reverse=True)
    top_evolution = top_evolution[:10]

    score_rows = player_avg_rows
    category_counts = {'Alto potencial': 0, 'Potencial medio': 0, 'Bajo potencial': 0, 'Sin datos': 0}
    players_with_stats = set()
    for player_id, avg_score in score_rows:
        players_with_stats.add(player_id)
        if avg_score is None:
            continue
        if avg_score >= 7.5:
            category_counts['Alto potencial'] += 1
        elif avg_score >= 5:
            category_counts['Potencial medio'] += 1
        else:
            category_counts['Bajo potencial'] += 1
    category_counts['Sin datos'] = max(total_players - len(players_with_stats), 0)
    final_score_avg = db.query(func.avg(PlayerStat.final_score)).filter(PlayerStat.final_score != None).scalar()
    final_score_avg = round(float(final_score_avg), 2) if final_score_avg is not None else None
    db.close()

    pot_labels = list(category_counts.keys())
    pot_values = [category_counts[label] for label in pot_labels]
    html = render_template(
        'dashboard.html',
        positions=positions,
        pos_values=pos_values,
        pot_labels=pot_labels,
        pot_values=pot_values,
        avg_labels=avg_labels,
        avg_values=avg_values,
        total_players=total_players,
        final_score_avg=final_score_avg,
        top_potential=top_potential,
        top_evolution=top_evolution,
        selected_period=period,
        start_date_str=start_str,
        end_date_str=end_str,
    )
    _cache_set(dashboard_cache_key, html)
    return html
# ----------------------------------------------------
# LISTA JUGADORES
@app.route("/players")
@login_required
def index():
    search_term = request.args.get('q')
    pos_filter = request.args.get('position')
    club_filter = request.args.get('club')
    country_filter = request.args.get('country')
    top_potential = request.args.get('top_potential')
    order_attr = request.args.get('order_attr')
    page = request.args.get('page', 1, type=int)
    per_page = 50
    db = Session()
    query = db.query(Player).options(joinedload(Player.stats))
    pos_list = [r[0] for r in db.query(Player.position).distinct().all()]
    club_list = [r[0] for r in db.query(Player.club).distinct().all() if r[0]]
    country_list = [r[0] for r in db.query(Player.country).distinct().all() if r[0]]
    if search_term:
        query = query.filter(Player.name.ilike(f"%{search_term}%"))
    if pos_filter:
        query = query.filter(Player.position == pos_filter)
    if club_filter:
        query = query.filter(Player.club == club_filter)
    if country_filter:
        query = query.filter(Player.country == country_filter)
    if top_potential:
        query = query.filter(Player.potential_label.is_(True))
    if order_attr and hasattr(Player, order_attr):
        query = query.order_by(desc(getattr(Player, order_attr)))
    total = query.count()
    total_pages = (total + per_page - 1) // per_page
    players = query.offset((page - 1) * per_page).limit(per_page).all()
    if top_potential:
        # Para el filtro de alto potencial, ordenar por potencial real (mayor->menor)
        # y paginar luego del ordenamiento para que sea consistente entre páginas.
        players = query.all()
    player_rows = []
    for player in players:
        projection = compute_projection(player, db_session=db)
        attr_map = player_attribute_map(player)
        best_position, best_score = recommend_position_from_attrs(attr_map)
        fit_score = weighted_score_from_attrs(attr_map, player.position)
        if projection:
            combined_pct = projection["combined_prob"] * 100
            row = {
                "player": player,
                "photo_url": player.photo_url or default_player_photo_url(
                    name=player.name,
                    national_id=player.national_id,
                    fallback=str(player.id),
                ),
                "category": projection["category"],
                "probability": f"{combined_pct:.1f}%",
                "prob_value": combined_pct,
                "fit_score": projection.get("fit_score", fit_score),
                "best_position": projection.get("recommended_position", best_position),
                "best_score": projection.get("recommended_score", best_score),
            }
        else:
            row = {
                "player": player,
                "photo_url": player.photo_url or default_player_photo_url(
                    name=player.name,
                    national_id=player.national_id,
                    fallback=str(player.id),
                ),
                "category": "Sin datos suficientes",
                "probability": "--",
                "prob_value": None,
                "fit_score": fit_score,
                "best_position": best_position,
                "best_score": best_score,
            }
        player_rows.append(row)

    if top_potential:
        player_rows.sort(
            key=lambda item: (item["prob_value"] is not None, item["prob_value"] or -1.0),
            reverse=True,
        )
        total = len(player_rows)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = min(max(page, 1), total_pages)
        start = (page - 1) * per_page
        end = start + per_page
        player_rows = player_rows[start:end]
    db.close()
    return render_template("players.html",
                           players=player_rows,
                           search_term=search_term,
                           pos_list=pos_list, club_list=club_list, country_list=country_list,
                           pos_filter=pos_filter, club_filter=club_filter, country_filter=country_filter,
                           top_potential=top_potential, order_attr=order_attr,
                           page=page, total_pages=total_pages, total_results=total)

# ----------------------------------------------------
# DETALLE
@app.route("/player/<int:player_id>")
@login_required
def player_detail(player_id: int):
    db = Session()
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        db.close()
        abort(404)
    history_synced = sync_player_attribute_history(player, db, note="Sincronizacion automatica al abrir ficha")
    if history_synced:
        db.commit()
    stats = fetch_player_stats(player_id, db_session=db)
    recent_stats = list(reversed(stats[-3:])) if stats else []
    attr_history = fetch_attribute_history(player_id, db_session=db)
    recent_attributes = list(reversed(attr_history[-3:])) if attr_history else []
    attribute_summary = summarize_attribute_history(attr_history)
    stats_summary = summarize_stats(stats)
    db.close()  # player_detail_db_closed
    player_photo_url = player.photo_url or default_player_photo_url(
        name=player.name,
        national_id=player.national_id,
        fallback=str(player.id),
    )
    projection = compute_projection(player, stats)
    attr_map = player_attribute_map(player)
    best_position, best_position_score = recommend_position_from_attrs(attr_map)
    current_fit = weighted_score_from_attrs(attr_map, player.position)
    position_ranking = [
        {
            "position": pos,
            "score": weighted_score_from_attrs(attr_map, pos),
            "is_current": normalized_position(player.position) == pos,
        }
        for pos in POSITION_OPTIONS
    ]
    position_ranking.sort(key=lambda item: item["score"], reverse=True)
    technical_attributes = [
        {"label": "Ritmo", "value": player.pace},
        {"label": "Disparo", "value": player.shooting},
        {"label": "Pase", "value": player.passing},
        {"label": "Regate", "value": player.dribbling},
        {"label": "Defensa", "value": player.defending},
        {"label": "Físico", "value": player.physical},
        {"label": "Visión", "value": player.vision},
        {"label": "Marcaje", "value": player.tackling},
        {"label": "Determinación", "value": player.determination},
        {"label": "Técnica", "value": player.technique},
    ]

    def build_trait(name: str, score: float, strengths: str, follow_up: str, improvement: str) -> dict:
        band = score_band(score)
        messaging = {"Alto": strengths, "Medio": follow_up, "Bajo": improvement}
        return {
            "name": name,
            "score": round(score, 1),
            "band": band,
            "message": messaging[band],
        }

    psychological_profile = [
        build_trait(
            "Resiliencia competitiva",
            player.determination,
            "Sostiene el esfuerzo bajo presión; indicado para partidos decisivos.",
            "Trabajar rutinas de respiración y feedback constante para fortalecer su respuesta en escenarios adversos.",
            "Recomendar intervención del área psicológica y refuerzo en hábitos de disciplina diaria.",
        ),
        build_trait(
            "Visión táctica",
            (player.vision + player.passing) / 2,
            "Lee espacios y acelera cambios de juego, facilita la progresión del equipo.",
            "Incrementar análisis de vídeo para mejorar la toma de decisiones en el último tercio.",
            "Diseñar ejercicios de toma de decisiones en superioridad/inferioridad numérica.",
        ),
        build_trait(
            "Creatividad ofensiva",
            (player.technique + player.dribbling) / 2,
            "Desborde y control diferenciales; puede romper líneas defensivas.",
            "Trabajar gestos técnicos específicos a alta velocidad para trasladar virtudes al contexto profesional.",
            "Enfocar sesiones en conducción orientada y confianza en el uno contra uno.",
        ),
        build_trait(
            "Liderazgo comunicacional",
            (player.vision + player.determination + player.passing) / 3,
            "Influye en sus compañeros y ordena fases ofensivas.",
            "Definir responsabilidades puntuales dentro del equipo para ganar protagonismo progresivo.",
            "Establecer mentoría con referentes del plantel y dinámicas de comunicación en cancha.",
        ),
    ]

    development_focus = compute_suggestions(player, threshold=15, top_n=3)

    return render_template(
        "player_detail.html",
        player=player,
        player_photo_url=player_photo_url,
        technical_attributes=technical_attributes,
        radar_labels=[attr["label"] for attr in technical_attributes],
        radar_values=[attr["value"] for attr in technical_attributes],
        psychological_profile=psychological_profile,
        development_focus=development_focus,
        recent_stats=recent_stats,
        stats_summary=stats_summary,
        history_payload=stats_chart_payload(stats),
        attribute_history=recent_attributes,
        attribute_summary=attribute_summary,
        attribute_payload=attribute_chart_payload(attr_history),
        attribute_labels=ATTRIBUTE_LABELS,
        projection=projection,
        best_position=best_position,
        best_position_score=best_position_score,
        current_fit=current_fit,
        position_ranking=position_ranking[:3],
    )


@app.route("/player/<int:player_id>/stats", methods=["GET", "POST"])
@login_required
def player_stats(player_id: int):
    if request.method == "POST":
        _require_csrf()

    db = Session()
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        db.close()
        abort(404)

    if request.method == "POST":
        action = request.form.get("action", "add")
        if action == "recalculate":
            refresh_player_potential(player, db)
            db.commit()
            db.close()
            flash("Listo: se actualizo la proyeccion con los ultimos datos.", "success")
            return redirect(url_for("predict_player", player_id=player_id))

        try:
            record_date_str = request.form.get("record_date")
            record_date = datetime.strptime(record_date_str, "%Y-%m-%d").date() if record_date_str else date.today()
        except ValueError:
            record_date = date.today()

        def to_int(field: str) -> int:
            value = request.form.get(field)
            return int(value) if value not in (None, "",) else 0

        def to_float(field: str) -> Optional[float]:
            value = request.form.get(field)
            return float(value) if value not in (None, "",) else None

        stat = PlayerStat(
            player_id=player_id,
            record_date=record_date,
            matches_played=to_int("matches_played"),
            goals=to_int("goals"),
            assists=to_int("assists"),
            minutes_played=to_int("minutes_played"),
            yellow_cards=to_int("yellow_cards"),
            red_cards=to_int("red_cards"),
            pass_accuracy=to_float("pass_accuracy"),
            shot_accuracy=to_float("shot_accuracy"),
            duels_won_pct=to_float("duels_won_pct"),
            final_score=to_float("final_score"),
            notes=request.form.get("notes") or None,
        )
        if stat.final_score is None:
            stat.final_score = calculate_stats_rating(
                {
                    "matches": stat.matches_played,
                    "goals": stat.goals,
                    "assists": stat.assists,
                    "minutes": stat.minutes_played,
                    "pass_pct": stat.pass_accuracy,
                    "shot_pct": stat.shot_accuracy,
                    "duels_pct": stat.duels_won_pct,
                }
            )
        db.add(stat)
        db.commit()
        refresh_player_potential(player, db)
        db.commit()
        db.close()
        flash("Listo: se agrego el registro al historial del jugador.", "success")
        return redirect(url_for("player_stats", player_id=player_id))

    stats = (db.query(PlayerStat)
             .filter(PlayerStat.player_id == player_id)
             .order_by(PlayerStat.record_date.desc(), PlayerStat.id.desc())
             .all())
    db.close()
    summary = summarize_stats(list(reversed(stats)))
    return render_template(
        "player_stats.html",
        player=player,
        stats=stats,
        summary=summary,
    )

# ----------------------------------------------------
# HISTORIAL DE ATRIBUTOS
@app.route("/player/<int:player_id>/attributes", methods=["GET", "POST"])
@login_required
def player_attributes(player_id: int):
    if request.method == "POST":
        _require_csrf()

    db = Session()
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        db.close()
        abort(404)

    if request.method == "POST":
        action = request.form.get("action", "add")
        if action == "recalculate":
            refresh_player_potential(player, db)
            db.commit()
            db.close()
            flash("Listo: se actualizo la proyeccion con los nuevos atributos.", "success")
            return redirect(url_for('predict_player', player_id=player_id))
        try:
            record_date_str = request.form.get("record_date")
            record_date = datetime.strptime(record_date_str, "%Y-%m-%d").date() if record_date_str else date.today()
        except ValueError:
            record_date = date.today()

        entry = PlayerAttributeHistory(player_id=player_id, record_date=record_date, notes=request.form.get("notes") or None)
        for field in ATTRIBUTE_FIELDS:
            raw = request.form.get(field)
            value = int(raw) if raw not in (None, "") else None
            setattr(entry, field, value)
            if value is not None:
                setattr(player, field, value)
        db.add(entry)
        refresh_player_potential(player, db)
        db.commit()
        db.close()
        flash("Listo: se guardo el historial de atributos.", "success")
        return redirect(url_for("player_attributes", player_id=player_id))

    history = (db.query(PlayerAttributeHistory)
               .filter(PlayerAttributeHistory.player_id == player_id)
               .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
               .all())
    ascending_history = list(reversed(history))
    summary = summarize_attribute_history(ascending_history)
    payload = attribute_chart_payload(ascending_history)
    db.close()
    return render_template(
        "player_attributes.html",
        player=player,
        history=history,
        summary=summary,
        attribute_labels=ATTRIBUTE_LABELS,
        payload=payload,
    )

# ----------------------------------------------------
# EDITAR JUGADOR
@app.route('/edit_player/<int:player_id>', methods=['GET', 'POST'])
@login_required
def edit_player(player_id):
    if request.method == "POST":
        _require_csrf()

    db = Session()
    player = db.get(Player, player_id)
    if not player:
        db.close()
        abort(404)
    if request.method == 'POST':
        errors: List[str] = []
        name = (request.form.get('name') or '').strip()
        national_id = normalize_identifier(request.form.get("national_id"))
        age = parse_int_field(request.form.get('age'))
        position = normalize_position_choice(request.form.get('position'))
        club = (request.form.get('club') or '').strip() or None
        country = (request.form.get('country') or '').strip() or None
        photo_url = (request.form.get('photo_url') or '').strip() or None
        if not name:
            errors.append("El nombre es obligatorio.")
        if not national_id:
            errors.append("El DNI debe contener solo n���meros.")
        else:
            repeated = (
                db.query(Player.id)
                .filter(Player.national_id == national_id, Player.id != player.id)
                .first()
            )
            if repeated:
                errors.append("El DNI ingresado pertenece a otro jugador.")
        if not is_valid_eval_age(age):
            errors.append("La edad debe estar entre 12 y 18 a���os.")
        attr_values: Dict[str, int] = {}
        for field in ATTRIBUTE_FIELDS:
            value = parse_int_field(request.form.get(field), getattr(player, field))
            if not is_valid_attribute(value):
                errors.append(f"{ATTRIBUTE_LABELS[field]} debe estar entre 0 y 20.")
            attr_values[field] = value
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return render_template('edit_player.html', player=player, form_data=request.form, position_options=POSITION_OPTIONS)
        player.name = name
        player.national_id = national_id
        player.age = age
        player.position = position
        player.club = club
        player.country = country
        player.photo_url = photo_url or default_player_photo_url(
            name=name,
            national_id=national_id,
            fallback=str(player.id),
        )
        for field, value in attr_values.items():
            setattr(player, field, value)
        player.potential_label = True if request.form.get('potential_label') == '1' else False
        sync_player_attribute_history(player, db, note="Actualizacion de ficha")
        refresh_player_potential(player, db)
        db.commit()
        db.close()
        flash("Listo: se actualizo la ficha del jugador.", "success")
        return redirect(url_for('player_detail', player_id=player_id))
    db.close()
    return render_template('edit_player.html', player=player, position_options=POSITION_OPTIONS)

# ----------------------------------------------------
# ELIMINAR JUGADOR
@app.route('/delete_player/<int:player_id>', methods=['POST'])
@login_required
def delete_player(player_id):
    _require_csrf()

    db = Session()
    player = db.get(Player, player_id)
    if not player:
        db.close()
        abort(404)
    db.delete(player)
    db.commit()
    db.close()
    flash("Listo: se elimino el jugador del seguimiento.", "success")
    return redirect(url_for('index'))

@app.route("/player/<int:player_id>/predict")
@login_required
def predict_player(player_id: int):
    if model is None:
        return "No hay calculo disponible todavia. Actualiza los datos desde Configuracion.", 500

    db = Session()
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        db.close()
        abort(404)
    projection = refresh_player_potential(player, db)
    player_view = SimpleNamespace(**player.to_dict())
    player_view.photo_url = player.photo_url or default_player_photo_url(
        name=player.name,
        national_id=player.national_id,
        fallback=str(player.id),
    )
    player_view.potential_label = player.potential_label
    suggestions = compute_suggestions(player_view)
    attr_map = player_attribute_map(player)
    best_position, best_position_score = recommend_position_from_attrs(attr_map)
    current_fit = weighted_score_from_attrs(attr_map, player.position)

    if not projection:
        stats_summary = summarize_stats([])
        prob_base = prob_combined = 0.0
        category = "Sin datos suficientes"
        stats_history = []
    else:
        prob_base = projection["base_prob"]
        prob_combined = projection["combined_prob"]
        category = projection["category"]
        stats_summary = projection["stats_summary"]
        stats_history = projection["history"]
    history_payload = stats_chart_payload(stats_history)
    attribute_payload = attribute_chart_payload(fetch_attribute_history(player_id))
    db.commit()
    db.close()

    return render_template(
        "prediction.html",
        player=player_view,
        probability=f"{prob_combined*100:.1f}%",
        probability_base=f"{prob_base*100:.1f}%",
        probability_delta=(prob_combined - prob_base) * 100,
        category=category,
        suggestions=suggestions,
        stats_summary=stats_summary,
        history_payload=history_payload,
        attribute_payload=attribute_payload,
        attribute_labels=ATTRIBUTE_LABELS,
        best_position=best_position,
        best_position_score=best_position_score,
        current_fit=current_fit,
    )
# ----------------------------------------------------
# COMPARADORES
@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare_players():
    db = Session()

    # Limitar la cantidad de jugadores que se cargan en el combo
    MAX_COMPARE_PLAYERS = 2000
    rows = (
        db.query(Player.id, Player.name, Player.position)
        .order_by(Player.name.asc())
        .limit(MAX_COMPARE_PLAYERS + 1)
        .all()
    )
    truncated = len(rows) > MAX_COMPARE_PLAYERS
    if truncated:
        rows = rows[:MAX_COMPARE_PLAYERS]

    # Usamos SimpleNamespace para tener player.id / player.name / player.position
    players = [
        SimpleNamespace(id=row[0], name=row[1], position=row[2])
        for row in rows
    ]

    selected_one = None
    selected_two = None
    comparison = None
    target_position = None

    if request.method == "POST":
        selected_one = request.form.get("player_one", type=int)
        selected_two = request.form.get("player_two", type=int)
        target_position_raw = request.form.get("target_position")
        target_position = normalized_position(target_position_raw) if target_position_raw else None

        if selected_one and selected_two and selected_one != selected_two:
            # Traemos SOLO los dos jugadores seleccionados
            selected = (
                db.query(Player)
                .filter(Player.id.in_([selected_one, selected_two]))
                .all()
            )
            players_map = {p.id: p for p in selected}
            player_one = players_map.get(selected_one)
            player_two = players_map.get(selected_two)

            if player_one and player_two:
                # Stats solo de estos dos jugadores
                stats_one = fetch_player_stats(player_one.id, db_session=db)
                stats_two = fetch_player_stats(player_two.id, db_session=db)

                summary_one = summarize_stats(stats_one)
                summary_two = summarize_stats(stats_two)

                # Proyección solo para estos dos
                projection_one = compute_projection(
                    player_one, stats=stats_one, db_session=db
                )
                projection_two = compute_projection(
                    player_two, stats=stats_two, db_session=db
                )

                prob_one = projection_one["combined_prob"] if projection_one else 0.0
                prob_two = projection_two["combined_prob"] if projection_two else 0.0

                # Comparación atributo por atributo
                attr_rows = []
                score_one = 0
                score_two = 0

                base_position = target_position or normalized_position(player_one.position or player_two.position)
                weights_map = position_weights(base_position)

                for field in ATTRIBUTE_FIELDS:
                    label = ATTRIBUTE_LABELS[field]
                    value_one = getattr(player_one, field)
                    value_two = getattr(player_two, field)
                    weight = weights_map.get(field, 0.0)

                    if value_one > value_two:
                        winner = 1
                        score_one += weight
                    elif value_two > value_one:
                        winner = 2
                        score_two += weight
                    else:
                        winner = 0

                    attr_rows.append(
                        {
                            "label": label,
                            "value_one": value_one,
                            "value_two": value_two,
                            "weight": weight,
                            "winner": winner,
                        }
                    )

                avg_one = summary_one.get("avg_final_score")
                avg_two = summary_two.get("avg_final_score")

                attr_map_one = player_attribute_map(player_one)
                attr_map_two = player_attribute_map(player_two)
                total_one = weighted_score_from_attrs(attr_map_one, base_position) + (avg_one or 0)
                total_two = weighted_score_from_attrs(attr_map_two, base_position) + (avg_two or 0)
                best_pos_one, best_pos_one_score = recommend_position_from_attrs(attr_map_one)
                best_pos_two, best_pos_two_score = recommend_position_from_attrs(attr_map_two)
                fit_one = weighted_score_from_attrs(attr_map_one, base_position)
                fit_two = weighted_score_from_attrs(attr_map_two, base_position)

                if total_one > total_two:
                    conclusion = (
                        f"{player_one.name} presenta mejores indicadores generales "
                        f"respecto a {player_two.name}."
                    )
                elif total_two > total_one:
                    conclusion = (
                        f"{player_two.name} presenta mejores indicadores generales "
                        f"respecto a {player_one.name}."
                    )
                else:
                    conclusion = (
                        "Ambos jugadores presentan indicadores equivalentes "
                        "en la comparación."
                    )

                comparison = {
                    "player_one": {
                        "name": player_one.name,
                        "position": player_one.position,
                        "probability": prob_one,
                        "avg_score": avg_one,
                        "fit_score": fit_one,
                        "best_position": best_pos_one,
                        "best_position_score": best_pos_one_score,
                    },
                    "player_two": {
                        "name": player_two.name,
                        "position": player_two.position,
                        "probability": prob_two,
                        "avg_score": avg_two,
                        "fit_score": fit_two,
                        "best_position": best_pos_two,
                        "best_position_score": best_pos_two_score,
                    },
                    "attributes": attr_rows,
                    "score_one": score_one,
                    "score_two": score_two,
                    "conclusion": conclusion,
                    "target_position": base_position,
                }

    db.close()
    return render_template(
        "compare.html",
        players=players,
        selected_one=selected_one,
        selected_two=selected_two,
        comparison=comparison,
        truncated=truncated,
        max_players=MAX_COMPARE_PLAYERS,
        target_position=target_position,
    )


@app.route("/compare/multi", methods=["GET", "POST"])
@login_required
def compare_multi():
    db = Session()

    # Igual que en el comparador 1vs1: limitamos la lista de jugadores
    MAX_COMPARE_PLAYERS = 2000
    rows = (
        db.query(Player.id, Player.name, Player.position)
        .order_by(Player.name.asc())
        .limit(MAX_COMPARE_PLAYERS + 1)
        .all()
    )
    truncated = len(rows) > MAX_COMPARE_PLAYERS
    if truncated:
        rows = rows[:MAX_COMPARE_PLAYERS]

    players = [
        SimpleNamespace(id=row[0], name=row[1], position=row[2])
        for row in rows
    ]

    selected_ids: List[int] = []
    comparison = None
    target_position = None

    # Ranking global por puesto (4 pestañas): Arquero, Defensa, Mediocampista, Delantero.
    # Nota: Lateral se agrupa en Defensa para simplificar lectura táctica.
    position_tabs = [
        {"key": "arquero", "label": "Arqueros", "bucket": "Portero"},
        {"key": "defensa", "label": "Defensas", "bucket": "Defensa"},
        {"key": "mediocampo", "label": "Mediocampistas", "bucket": "Mediocampista"},
        {"key": "delantera", "label": "Delanteros", "bucket": "Delantero"},
    ]
    ranking_by_position: Dict[str, List[Dict[str, object]]] = {tab["key"]: [] for tab in position_tabs}

    all_players_for_ranking = db.query(Player).all()
    for player in all_players_for_ranking:
        projection = compute_projection(player, db_session=db)
        if not projection:
            continue
        normalized_pos = normalized_position(player.position)
        if normalized_pos == "Lateral":
            normalized_pos = "Defensa"

        tab = next((tab for tab in position_tabs if tab["bucket"] == normalized_pos), None)
        if not tab:
            continue

        ranking_by_position[tab["key"]].append(
            {
                "id": player.id,
                "name": player.name,
                "position": player.position,
                "club": player.club,
                "age": player.age,
                "probability": round(float(projection["combined_prob"]) * 100, 1),
                "category": projection["category"],
            }
        )

    for tab in position_tabs:
        rows = ranking_by_position[tab["key"]]
        rows.sort(key=lambda item: item["probability"], reverse=True)
        ranking_by_position[tab["key"]] = rows[:10]

    if request.method == "POST":
        raw_ids = request.form.getlist("players")
        try:
            selected_ids = [int(pid) for pid in raw_ids][:10]
        except ValueError:
            selected_ids = []
        target_position_raw = request.form.get("target_position")
        target_position = normalized_position(target_position_raw) if target_position_raw else None

        if selected_ids:
            selected_players = (
                db.query(Player)
                .filter(Player.id.in_(selected_ids))
                .order_by(Player.name.asc())
                .all()
            )

            if selected_players:
                # Promedios de final_score solo para estos jugadores (se usa para el total)
                score_rows = (
                    db.query(PlayerStat.player_id, func.avg(PlayerStat.final_score))
                    .filter(
                        PlayerStat.player_id.in_([p.id for p in selected_players])
                    )
                    .group_by(PlayerStat.player_id)
                    .all()
                )
                avg_score_map = {
                    player_id: (float(avg) if avg is not None else None)
                    for player_id, avg in score_rows
                }

                # Atributos a graficar (eran los del radar)
                labels = [ATTRIBUTE_LABELS[field] for field in ATTRIBUTE_FIELDS]

                # Datasets para gráfico de barras (uno por jugador)
                datasets = []
                colors = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                ]

                ranking: List[dict] = []

                base_position = target_position
                for idx, player in enumerate(selected_players):
                    values = [getattr(player, field) for field in ATTRIBUTE_FIELDS]
                    color = colors[idx % len(colors)]

                    datasets.append(
                        {
                            "label": player.name,
                            "data": values,
                            "backgroundColor": color,
                            "borderColor": color,
                            "borderWidth": 1,
                        }
                    )

                    # Puntaje total = suma atributos + promedio histórico (si existe)
                    attr_map = player_attribute_map(player)
                    attribute_sum = weighted_score_from_attrs(attr_map, base_position or player.position)
                    avg_score = avg_score_map.get(player.id)
                    total = attribute_sum + (avg_score or 0)

                    # Mapa atributo -> valor (con los labels “bonitos”)
                    attributes_map = {
                        ATTRIBUTE_LABELS[f]: getattr(player, f) for f in ATTRIBUTE_FIELDS
                    }

                    ranking.append(
                        {
                            "name": player.name,
                            "position": player.position,
                            "attributes_map": attributes_map,
                            "total": round(total, 2),
                        }
                    )

                # Ordenamos ranking por total (desc)
                ranking.sort(key=lambda item: item["total"], reverse=True)

                # Gráfico de barras horizontal con promedios históricos (igual que antes)
                score_labels = [r["name"] for r in ranking]
                # Para no mostrar la columna "promedio" en la tabla, lo usamos solo en el gráfico
                score_values = []
                for r in ranking:
                    # reconstruimos avg desde attributes_map+suma si quisiéramos, pero ya lo tenemos en avg_score_map
                    # buscamos por nombre
                    p = next((p for p in selected_players if p.name == r["name"]), None)
                    avg_val = avg_score_map.get(p.id) if p else None
                    score_values.append(avg_val or 0)

                comparison = {
                    "chart_payload": {
                        "labels": labels,
                        "datasets": datasets,
                    },
                    "score_payload": {
                        "labels": score_labels,
                        "datasets": [
                            {
                                "label": "Promedio historial",
                                "data": score_values,
                                "backgroundColor": "rgba(13, 110, 253, 0.6)",
                                "borderColor": "#0d6efd",
                            }
                        ],
                    },
                    "ranking": ranking,   # ahora trae attributes_map y total
                    "target_position": base_position,
                }

    db.close()
    return render_template(
        "compare_multi.html",
        players=players,
        selected_ids=selected_ids,
        comparison=comparison,
        truncated=truncated,
        max_players=MAX_COMPARE_PLAYERS,
        target_position=target_position,
        position_tabs=position_tabs,
        ranking_by_position=ranking_by_position,
    )




# ----------------------------------------------------
# CRUD COACHES (igual estrategia)
@app.route("/coaches")
@login_required
def list_coaches():
    db = Session()
    coaches = db.query(Coach).all()
    db.close()
    return render_template("coaches.html", coaches=coaches)

@app.route("/coaches/new", methods=["GET", "POST"])
@login_required
def new_coach():
    if request.method == "POST":
        _require_csrf()

    if request.method == "POST":
        db = Session()
        coach = Coach(
            name=request.form["name"],
            role=request.form["role"],
            age=request.form.get("age"),
            club=request.form.get("club"),
            country=request.form.get("country")
        )
        db.add(coach)
        db.commit()
        db.close()
        return redirect(url_for("list_coaches"))
    return render_template("coach_form.html", coach=None)

@app.route("/coaches/edit/<int:coach_id>", methods=["GET", "POST"])
@login_required
def edit_coach(coach_id):
    if request.method == "POST":
        _require_csrf()

    db = Session()
    coach = db.query(Coach).get(coach_id)
    if not coach:
        db.close()
        abort(404)
    if request.method == "POST":
        coach.name = request.form["name"]
        coach.role = request.form["role"]
        coach.age = request.form.get("age")
        coach.club = request.form.get("club")
        coach.country = request.form.get("country")
        db.commit()
        db.close()
        return redirect(url_for("list_coaches"))
    db.close()
    return render_template("coach_form.html", coach=coach)

@app.route("/coaches/delete/<int:coach_id>", methods=["POST"])
@login_required
def delete_coach(coach_id):
    _require_csrf()

    db = Session()
    coach = db.query(Coach).get(coach_id)
    if not coach:
        db.close()
        abort(404)
    db.delete(coach)
    db.commit()
    db.close()
    return redirect(url_for("list_coaches"))


# ----------------------------------------------------
# CRUD DIRECTORES
@app.route("/directors")
@login_required
def list_directors():
    db = Session()
    directors = db.query(Director).all()
    db.close()
    return render_template("directors.html", directors=directors)


@app.route("/directors/new", methods=["GET", "POST"])
@login_required
def new_director():
    if request.method == "POST":
        _require_csrf()

    if request.method == "POST":
        db = Session()
        director = Director(
            name=request.form["name"],
            position=request.form["position"],
            age=request.form.get("age"),
            club=request.form.get("club"),
            country=request.form.get("country"),
        )
        db.add(director)
        db.commit()
        db.close()
        return redirect(url_for("list_directors"))
    return render_template("director_form.html", director=None)


@app.route("/directors/edit/<int:director_id>", methods=["GET", "POST"])
@login_required
def edit_director(director_id):
    if request.method == "POST":
        _require_csrf()

    db = Session()
    director = db.query(Director).get(director_id)
    if not director:
        db.close()
        abort(404)
    if request.method == "POST":
        director.name = request.form["name"]
        director.position = request.form["position"]
        director.age = request.form.get("age")
        director.club = request.form.get("club")
        director.country = request.form.get("country")
        db.commit()
        db.close()
        return redirect(url_for("list_directors"))
    db.close()
    return render_template("director_form.html", director=director)


@app.route("/directors/delete/<int:director_id>", methods=["POST"])
@login_required
def delete_director(director_id):
    _require_csrf()

    db = Session()
    director = db.query(Director).get(director_id)
    if not director:
        db.close()
        abort(404)
    db.delete(director)
    db.commit()
    db.close()
    return redirect(url_for("list_directors"))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    # CSRF mínimo para POST
    if request.method == "POST":
        _require_csrf()

    # Admin-only
    if session.get("role") != "administrador":
        abort(403)

    status_messages: List[str] = []
    modal_message: Optional[str] = None

    if request.method == "POST":
        action = request.form.get("action")
        if action == "update_database":
            start = time.time()
            if not _PIPELINE_LOCK.acquire(blocking=False):
                status_messages = ["Ya hay una actualizacion en curso. Reintenta en unos minutos."]
                flash("Ya hay una actualizacion en curso. Reintenta en unos minutos.", "warning")
            else:
                try:
                    success, logs = update_database_pipeline(
                        limit=EVAL_POOL_MAX,
                        sync_shortlist=SYNC_SHORTLIST_ENABLED,
                    )
                    status_messages = logs
                finally:
                    duration = round(time.time() - start, 2)
                    status_messages.append(f"Duracion total de la actualizacion: {duration}s")
                    try:
                        app.logger.info("Pipeline update_database finished in %ss", duration)
                    except Exception:
                        pass
                    _PIPELINE_LOCK.release()

                if success:
                    if SYNC_SHORTLIST_ENABLED:
                        modal_message = "Se actualizaron puntajes y se sincronizo la base operativa."
                    else:
                        modal_message = "Se actualizaron puntajes (sin sincronizar jugadores operativos)."
                    flash("Listo: la actualizacion general finalizo correctamente.", "success")
                else:
                    flash("No se pudo completar la actualizacion. Revisa el detalle.", "danger")

    return render_template(
        "settings.html",
        status_messages=status_messages,
        modal_message=modal_message,
    )



def parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    value = value.strip().lower()
    return value in {"1", "true", "t", "yes", "y", "si", "sí", "alto"}


def parse_int_field(value: Optional[str], default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_position_choice(value: Optional[str]) -> str:
    return normalized_position(value)


@app.route("/players/manage", methods=["GET", "POST"])
@login_required
def manage_players():
    if request.method == "POST":
        _require_csrf()

    if request.method == "POST":
        mode = request.form.get("mode", "single")
        db = Session()
        created: List[str] = []
        errors: List[str] = []
        try:
            if mode == "single":
                current_total = db.query(func.count(Player.id)).scalar() or 0
                if current_total >= EVAL_POOL_MAX:
                    errors.append(
                        f"Se alcanzo el maximo de {EVAL_POOL_MAX} jugadores evaluables. "
                        "Elimina o actualiza jugadores existentes."
                    )
                name = (request.form.get("name") or "").strip()
                national_id = normalize_identifier(request.form.get("national_id"))
                age = parse_int_field(request.form.get("age"))
                position = normalize_position_choice(request.form.get("position"))
                club = (request.form.get("club") or "").strip() or None
                country = (request.form.get("country") or "").strip() or None
                photo_url = (request.form.get("photo_url") or "").strip() or None
                if not name:
                    errors.append("El nombre es obligatorio.")
                if not national_id:
                    errors.append("Ingresa un DNI/ID valido (solo numeros).")
                else:
                    repeated = (
                        db.query(Player.id)
                        .filter(Player.national_id == national_id)
                        .first()
                    )
                    if repeated:
                        errors.append("Ese DNI ya esta registrado en la base.")
                if not is_valid_eval_age(age):
                    errors.append("La edad debe estar entre 12 y 18 anos.")
                attr_values: Dict[str, int] = {}
                for field in ATTRIBUTE_FIELDS:
                    value = parse_int_field(request.form.get(field), 10)
                    if not is_valid_attribute(value):
                        errors.append(f"{ATTRIBUTE_LABELS[field]} debe ubicarse entre 0 y 20.")
                    attr_values[field] = value
                if not errors:
                    player = Player(
                        name=name,
                        national_id=national_id,
                        age=age,
                        position=position,
                        club=club,
                        country=country,
                        photo_url=photo_url or default_player_photo_url(name=name, national_id=national_id),
                        potential_label=parse_bool(request.form.get("potential_label")),
                        **attr_values,
                    )
                    db.add(player)
                    db.flush()
                    sync_player_attribute_history(player, db, note="Alta de jugador")
                    db.commit()
                    created.append(player.name)
                    flash(f"Listo: se agrego a {player.name} al seguimiento.", "success")
                    return redirect(url_for("manage_players"))
            elif mode == "bulk":
                bulk_input = request.form.get("bulk_input") or ""
                lines = [line.strip() for line in bulk_input.splitlines() if line.strip()]
                current_total = db.query(func.count(Player.id)).scalar() or 0
                available_slots = max(EVAL_POOL_MAX - current_total, 0)
                if available_slots <= 0:
                    errors.append(
                        f"No hay cupo disponible. Limite actual: {EVAL_POOL_MAX} jugadores evaluables."
                    )
                if not lines:
                    errors.append("Ingresa al menos una fila en la carga masiva.")
                else:
                    seen_ids = set()
                    for idx, line in enumerate(lines, start=1):
                        if len(created) >= available_slots:
                            errors.append(
                                f"Se alcanzo el limite de {EVAL_POOL_MAX} jugadores. "
                                "El resto de filas fue omitido."
                            )
                            break
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) < 17:
                            errors.append(f"Linea {idx}: se esperaban 17 o 18 columnas (recibido {len(parts)}).")
                            continue
                        try:
                            name = parts[0]
                            national_id = normalize_identifier(parts[1])
                            age = int(parts[2])
                            position = normalize_position_choice(parts[3])
                            club = parts[4] or None
                            country = parts[5] or None
                            raw_photo_url = parts[17] if len(parts) > 17 else ""
                            if not national_id:
                                raise ValueError("DNI no valido.")
                            if national_id in seen_ids:
                                raise ValueError("DNI repetido en la carga.")
                            exists = (
                                db.query(Player.id)
                                .filter(Player.national_id == national_id)
                                .first()
                            )
                            if exists:
                                raise ValueError("DNI ya registrado en la base.")
                            if not is_valid_eval_age(age):
                                raise ValueError("Edad fuera del rango permitido (12-18).")
                            attr_values = {}
                            for offset, field in enumerate(ATTRIBUTE_FIELDS, start=6):
                                value = int(parts[offset])
                                if not is_valid_attribute(value):
                                    raise ValueError(f"{ATTRIBUTE_LABELS[field]} fuera de rango.")
                                attr_values[field] = value
                            potential_flag = parse_bool(parts[16]) if len(parts) > 16 else False
                            player = Player(
                                name=name,
                                national_id=national_id,
                                age=age,
                                position=position,
                                club=club,
                                country=country,
                                photo_url=(raw_photo_url.strip() or default_player_photo_url(name=name, national_id=national_id)),
                                potential_label=potential_flag,
                                **attr_values,
                            )
                            seen_ids.add(national_id)
                        except (ValueError, IndexError) as exc:
                            errors.append(f"Linea {idx}: {exc}")
                            continue
                        db.add(player)
                        db.flush()
                        sync_player_attribute_history(player, db, note="Alta masiva de jugador")
                        created.append(player.name)
                    if created and not errors:
                        db.commit()
                        flash(f"Listo: se agregaron {len(created)} jugadores.", "success")
                    elif created and errors:
                        db.commit()
                        flash(f"Se agregaron {len(created)} jugadores, pero quedaron advertencias para revisar.", "warning")
                    else:
                        db.rollback()
            else:
                errors.append("Modo de carga desconocido.")
        finally:
            db.close()
        if errors:
            for message in errors:
                flash(message, "danger")
        if created and mode == "bulk":
            return redirect(url_for("manage_players"))
    attribute_sequence = [(field, ATTRIBUTE_LABELS[field]) for field in ATTRIBUTE_FIELDS]
    db_metrics = Session()
    current_total = db_metrics.query(func.count(Player.id)).scalar() or 0
    db_metrics.close()
    return render_template("manage_players.html",
                           attribute_labels=ATTRIBUTE_LABELS,
                           attribute_sequence=attribute_sequence,
                           position_options=POSITION_OPTIONS,
                           current_total=current_total,
                           max_players=EVAL_POOL_MAX)



# --- Error handlers (observabilidad mínima) ---

@app.errorhandler(400)
def handle_400(e):
    app.logger.warning("400 Bad Request - path=%s user_id=%s", request.path, session.get("user_id"))
    return render_template("error.html", code=400, message="Solicitud inválida"), 400

@app.errorhandler(413)
def handle_413(e):
    app.logger.warning("413 Payload Too Large - path=%s user_id=%s", request.path, session.get("user_id"))
    return render_template("error.html", code=413, message="Request demasiado grande"), 413



@app.errorhandler(403)
def handle_403(e):
    app.logger.warning("403 Forbidden - path=%s user_id=%s role=%s", request.path, session.get("user_id"), session.get("role"))
    return render_template("error.html", code=403, message="Acceso denegado"), 403

@app.errorhandler(404)
def handle_404(e):
    app.logger.info("404 Not Found - path=%s user_id=%s", request.path, session.get("user_id"))
    return render_template("error.html", code=404, message="Recurso no encontrado"), 404

@app.errorhandler(500)
def handle_500(e):
    app.logger.exception("500 Internal Server Error - path=%s user_id=%s role=%s", request.path, session.get("user_id"), session.get("role"))
    return render_template("error.html", code=500, message="Error interno del servidor"), 500
if __name__ == "__main__":
    app.run(debug=True)
