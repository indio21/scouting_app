from typing import List, Tuple, Callable, Optional, Dict
from datetime import datetime, date, timedelta
from statistics import mean
from types import SimpleNamespace
from flask import Flask, render_template, redirect, url_for, request, session, flash, abort, jsonify
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker
import numpy as np
import torch
from models import Base, Player, Coach, Director, User, PlayerStat, PlayerAttributeHistory
from train_model import PlayerNet, normalize_features
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = "reemplazar-esta-clave"

DB_URL = "sqlite:///players_updated_v2.db"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(engine)

# ----------------------------------------------------
# Usuario administrador inicial
def init_admin_user(username="admin", password="admin"):
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


@app.route("/healthz")
def healthz():
    """Liveness para infraestructura (Render)."""
    return jsonify({"status": "ok"}), 200


@app.route("/health")
def health():
    """Readiness simple con chequeo de DB."""
    db = None
    try:
        db = Session()
        db.query(func.count(Player.id)).scalar()
        return jsonify({"status": "ok", "db": "up"}), 200
    except Exception as exc:
        return jsonify({"status": "degraded", "db": "down", "detail": str(exc)}), 503
    finally:
        if db is not None:
            db.close()

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
    input_dim = 11
    model = PlayerNet(input_dim=input_dim)
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

MODEL_PATH = "model.pt"
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    model = None
    print("Advertencia: modelo no encontrado.")

def prepare_input(player: Player) -> torch.Tensor:
    features = [
        player.age, player.pace, player.shooting, player.passing, player.dribbling,
        player.defending, player.physical, player.vision, player.tackling,
        player.determination, player.technique
    ]
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


ATTRIBUTE_FIELDS = [
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

ATTRIBUTE_LABELS = {
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


def combine_probability(base_prob: float, stats_summary: Dict[str, Optional[float]]) -> float:
    avg_score = stats_summary.get("avg_final_score")
    if avg_score is None:
        return base_prob
    # Normalizamos la puntuacion final (1-10) y mezclamos con el modelo
    rating_weight = min(max(avg_score / 10.0, 0.0), 1.0)
    combined = (base_prob * 0.7) + (rating_weight * 0.3)
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
    if probability > 0.7:
        return "Alto potencial"
    if probability > 0.4:
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
    combined = combine_probability(base_prob, stats_summary)
    return {
        "base_prob": base_prob,
        "combined_prob": combined,
        "category": categorize_probability(combined),
        "stats_summary": stats_summary,
        "history": stats_list,
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
    init_admin_user()
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
    positions = [row[0] or 'Sin posicion' for row in pos_rows]
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
    if players:
        features = np.array([
            [
                player.age,
                player.pace,
                player.shooting,
                player.passing,
                player.dribbling,
                player.defending,
                player.physical,
                player.vision,
                player.tackling,
                player.determination,
                player.technique,
            ]
            for player in players
        ], dtype=np.float32)
        features_norm = normalize_features(features)
        with torch.no_grad():
            probs_tensor = model(torch.tensor(features_norm))
        base_probs = probs_tensor.squeeze().numpy().tolist()
        for idx, player in enumerate(players):
            base_prob = float(base_probs[idx])
            stats_summary = {"avg_final_score": avg_score_map.get(player.id)}
            combined = combine_probability(base_prob, stats_summary)
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
    return render_template(
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
    query = db.query(Player)
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
        query = query.order_by(desc(Player.potential_label))
    if order_attr and hasattr(Player, order_attr):
        query = query.order_by(desc(getattr(Player, order_attr)))
    total = query.count()
    total_pages = (total + per_page - 1) // per_page
    players = query.offset((page - 1) * per_page).limit(per_page).all()
    player_rows = []
    for player in players:
        projection = compute_projection(player, db_session=db)
        if projection:
            combined_pct = projection["combined_prob"] * 100
            row = {
                "player": player,
                "category": projection["category"],
                "probability": f"{combined_pct:.1f}%",
                "prob_value": combined_pct,
            }
        else:
            row = {
                "player": player,
                "category": "Sin modelo",
                "probability": "--",
                "prob_value": None,
            }
        player_rows.append(row)
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
    db.close()
    stats = fetch_player_stats(player_id)
    recent_stats = list(reversed(stats[-3:])) if stats else []
    attr_history = fetch_attribute_history(player_id)
    recent_attributes = list(reversed(attr_history[-3:])) if attr_history else []
    attribute_summary = summarize_attribute_history(attr_history)
    stats_summary = summarize_stats(stats)
    projection = compute_projection(player, stats)
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
    )


@app.route("/player/<int:player_id>/stats", methods=["GET", "POST"])
@login_required
def player_stats(player_id: int):
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
            flash("Potencial actualizado con los nuevos datos", "success")
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
        flash("Registro agregado al historial", "success")
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
            flash("Potencial actualizado con los nuevos atributos", "success")
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
        flash("Historial de atributos actualizado", "success")
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
    db = Session()
    player = db.query(Player).get(player_id)
    if not player:
        db.close()
        abort(404)
    if request.method == 'POST':
        player.name = request.form['name']
        player.position = request.form['position']
        player.club = request.form['club']
        player.country = request.form['country']
        for f in ["pace","shooting","passing","dribbling","defending","physical","vision","tackling","determination","technique"]:
                val = request.form.get(f)
                if val is not None and val != "":
                    setattr(player, f, int(val))
        # etiqueta de potencial (checkbox)
        player.potential_label = True if request.form.get('potential_label') == '1' else False
        refresh_player_potential(player, db)
        db.commit()
        db.close()
        flash('Jugador actualizado con exito', 'success')
        return redirect(url_for('player_detail', player_id=player_id))
    db.close()
    return render_template('edit_player.html', player=player)

# ----------------------------------------------------
# ELIMINAR JUGADOR
@app.route('/delete_player/<int:player_id>', methods=['POST'])
@login_required
def delete_player(player_id):
    db = Session()
    player = db.query(Player).get(player_id)
    if not player:
        db.close()
        abort(404)
    db.delete(player)
    db.commit()
    db.close()
    flash('Jugador eliminado con exito', 'success')
    return redirect(url_for('index'))

@app.route("/player/<int:player_id>/predict")
@login_required
def predict_player(player_id: int):
    if model is None:
        return "Modelo no cargado. Entrene el modelo primero.", 500

    db = Session()
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        db.close()
        abort(404)
    projection = refresh_player_potential(player, db)
    player_view = SimpleNamespace(**player.to_dict())
    player_view.potential_label = player.potential_label
    suggestions = compute_suggestions(player_view)

    if not projection:
        stats_summary = summarize_stats([])
        prob_base = prob_combined = 0.0
        category = "Modelo no disponible"
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

    if request.method == "POST":
        selected_one = request.form.get("player_one", type=int)
        selected_two = request.form.get("player_two", type=int)

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

                for field in ATTRIBUTE_FIELDS:
                    label = ATTRIBUTE_LABELS[field]
                    value_one = getattr(player_one, field)
                    value_two = getattr(player_two, field)

                    if value_one > value_two:
                        winner = 1
                        score_one += 1
                    elif value_two > value_one:
                        winner = 2
                        score_two += 1
                    else:
                        winner = 0

                    attr_rows.append(
                        {
                            "label": label,
                            "value_one": value_one,
                            "value_two": value_two,
                            "winner": winner,
                        }
                    )

                avg_one = summary_one.get("avg_final_score")
                avg_two = summary_two.get("avg_final_score")

                total_one = sum(
                    getattr(player_one, field) for field in ATTRIBUTE_FIELDS
                ) + (avg_one or 0)
                total_two = sum(
                    getattr(player_two, field) for field in ATTRIBUTE_FIELDS
                ) + (avg_two or 0)

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
                    },
                    "player_two": {
                        "name": player_two.name,
                        "position": player_two.position,
                        "probability": prob_two,
                        "avg_score": avg_two,
                    },
                    "attributes": attr_rows,
                    "score_one": score_one,
                    "score_two": score_two,
                    "conclusion": conclusion,
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

    if request.method == "POST":
        raw_ids = request.form.getlist("players")
        try:
            selected_ids = [int(pid) for pid in raw_ids][:10]
        except ValueError:
            selected_ids = []

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
                    attribute_sum = sum(values)
                    avg_score = avg_score_map.get(player.id)
                    total = attribute_sum + (avg_score or 0)

                    # Mapa atributo -> valor (con los labels “bonitos”)
                    attributes_map = {
                        ATTRIBUTE_LABELS[f]: getattr(player, f) for f in ATTRIBUTE_FIELDS
                    }

                    ranking.append(
                        {
                            "name": player.name,
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
                }

    db.close()
    return render_template(
        "compare_multi.html",
        players=players,
        selected_ids=selected_ids,
        comparison=comparison,
        truncated=truncated,
        max_players=MAX_COMPARE_PLAYERS,
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
    db = Session()
    coach = db.query(Coach).get(coach_id)
    if not coach:
        db.close()
        abort(404)
    db.delete(coach)
    db.commit()
    db.close()
    return redirect(url_for("list_coaches"))


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


@app.route("/players/manage", methods=["GET", "POST"])
@login_required
def manage_players():
    if request.method == "POST":
        mode = request.form.get("mode", "single")
        db = Session()
        created = []
        errors = []
        try:
            if mode == "single":
                name = (request.form.get("name") or "").strip()
                age = parse_int_field(request.form.get("age"))
                position = (request.form.get("position") or "").strip()
                club = request.form.get("club")
                country = request.form.get("country")
                if not name or not position or age <= 0:
                    errors.append("Nombre, edad y posición son obligatorios.")
                else:
                    player = Player(
                        name=name,
                        age=age,
                        position=position,
                        club=club,
                        country=country,
                        pace=parse_int_field(request.form.get("pace"), 10),
                        shooting=parse_int_field(request.form.get("shooting"), 10),
                        passing=parse_int_field(request.form.get("passing"), 10),
                        dribbling=parse_int_field(request.form.get("dribbling"), 10),
                        defending=parse_int_field(request.form.get("defending"), 10),
                        physical=parse_int_field(request.form.get("physical"), 10),
                        vision=parse_int_field(request.form.get("vision"), 10),
                        tackling=parse_int_field(request.form.get("tackling"), 10),
                        determination=parse_int_field(request.form.get("determination"), 10),
                        technique=parse_int_field(request.form.get("technique"), 10),
                        potential_label=parse_bool(request.form.get("potential_label")),
                    )
                    db.add(player)
                    db.commit()
                    created.append(player.name)
                    flash(f"Jugador '{player.name}' agregado correctamente.", "success")
                    db.close()
                    return redirect(url_for("manage_players"))
            elif mode == "bulk":
                bulk_input = request.form.get("bulk_input") or ""
                lines = [line.strip() for line in bulk_input.splitlines() if line.strip()]
                if not lines:
                    errors.append("Ingrese al menos una fila en el cargado masivo.")
                else:
                    for idx, line in enumerate(lines, start=1):
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) < 16:
                            errors.append(f"Línea {idx}: se esperaban 16 columnas (recibido {len(parts)}).")
                            continue
                        try:
                            player = Player(
                                name=parts[0],
                                age=int(parts[1]),
                                position=parts[2],
                                club=parts[3] or None,
                                country=parts[4] or None,
                                pace=int(parts[5]),
                                shooting=int(parts[6]),
                                passing=int(parts[7]),
                                dribbling=int(parts[8]),
                                defending=int(parts[9]),
                                physical=int(parts[10]),
                                vision=int(parts[11]),
                                tackling=int(parts[12]),
                                determination=int(parts[13]),
                                technique=int(parts[14]),
                                potential_label=parse_bool(parts[15]) if len(parts) > 15 else False,
                            )
                        except (ValueError, IndexError):
                            errors.append(f"Línea {idx}: no se pudo convertir los datos numéricos.")
                            continue
                        db.add(player)
                        created.append(player.name)
                    if created:
                        db.commit()
                        flash(f"Se agregaron {len(created)} jugadores.", "success")
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
    return render_template("manage_players.html",
                           attribute_labels=ATTRIBUTE_LABELS,
                           attribute_sequence=attribute_sequence)

if __name__ == "__main__":
    app.run(debug=True)
