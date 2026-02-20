import csv
from datetime import datetime
"""Entrenamiento del modelo de predicción de potencial.

Este script carga los datos de jugadores desde la base de datos SQLite,
prepara las variables de entrada y las etiquetas, entrena una red
neuronal simple en PyTorch y guarda el modelo entrenado.  También
imprime la precisión en el conjunto de prueba.

Uso:
    python train_model.py --db-url sqlite:///players.db --model-out model.pt
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from models import Base, Player
from player_logic import position_vector, MODEL_MIN_AGE, MODEL_MAX_AGE

SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Definición de la red neuronal
class PlayerNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def ensure_player_columns(engine) -> None:
    with engine.connect() as conn:
        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(players)"))]
        if "national_id" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN national_id TEXT"))
        if "photo_url" not in columns:
            conn.execute(text("ALTER TABLE players ADD COLUMN photo_url TEXT"))


def load_data(db_url: str):
    """Carga los jugadores desde la base de datos y devuelve matrices X, y."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    players = session.query(Player).all()
    # Variables de entrada: seleccionamos atributos y edad.
    X = []
    y = []
    for p in players:
        features = [
            p.age,
            p.pace,
            p.shooting,
            p.passing,
            p.dribbling,
            p.defending,
            p.physical,
            p.vision,
            p.tackling,
            p.determination,
            p.technique,
        ]
        features.extend(position_vector(p.position))
        X.append(features)
        y.append(1 if p.potential_label else 0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normaliza las características entre 0 y 1."""
    X_norm = X.copy()
    ages = np.clip(X[:, 0], MODEL_MIN_AGE, MODEL_MAX_AGE)
    X_norm[:, 0] = (ages - MODEL_MIN_AGE) / (MODEL_MAX_AGE - MODEL_MIN_AGE)
    X_norm[:, 1:11] = X_norm[:, 1:11] / 20.0
    return X_norm


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 10, lr: float = 1e-3):
    """Entrena la red neuronal y devuelve el modelo entrenado."""
    X_norm = normalize_features(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = PlayerNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f"Época {epoch+1}/{epochs}, pérdida: {loss.item():.4f}")

    # Evaluación
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        preds_binary = (preds >= 0.5).float()
        accuracy = (preds_binary.eq(y_test_tensor)).float().mean().item()
    print(f"Precisión en el conjunto de prueba: {accuracy*100:.2f}%")

    # Métricas adicionales (clasificación binaria)
    # - y_true: labels reales (0/1)
    # - y_prob: probabilidad predicha (0..1)
    y_true = y_test_tensor.cpu().numpy().reshape(-1)
    y_prob = preds.cpu().numpy().reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = None
    
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = None
    
    try:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
    except Exception:
        f1 = precision = recall = None
        cm = None
    
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR-AUC: {pr_auc:.4f}")
    if f1 is not None:
        print(f"F1: {f1:.4f} | Precisión: {precision:.4f} | Recall: {recall:.4f}")
    if cm is not None:
        print("Confusion matrix:", cm)
    
    # Persistir corrida (CSV) - si existe log_experiment()
    try:
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "seed": SEED,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "epochs": int(epochs),
            "lr": float(lr),
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc) if roc_auc is not None else "",
            "pr_auc": float(pr_auc) if pr_auc is not None else "",
            "f1": float(f1) if f1 is not None else "",
            "precision": float(precision) if precision is not None else "",
            "recall": float(recall) if recall is not None else "",
            "model_path": os.path.join(os.path.dirname(__file__), "model.pt"),
        }
        log_experiment(row)
    except Exception:
        pass
    
    return model


def save_model(model: nn.Module, path: str):
    """Guarda el modelo en disco."""
    torch.save(model.state_dict(), path)


def main(db_url: str, model_out: str, epochs: int, lr: float):
    X, y = load_data(db_url)
    model = train_model(X, y, epochs=epochs, lr=lr)
    save_model(model, model_out)
    print(f"Modelo guardado en {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de scouting")
    parser.add_argument("--db-url", type=str, default="sqlite:///players_training.db", help="URL de la base de datos de entrenamiento")
    parser.add_argument("--model-out", type=str, default="model.pt", help="Ruta de salida del modelo")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje")
    args = parser.parse_args()
    main(args.db_url, args.model_out, args.epochs, args.lr)

# --- Logging reproducible de experimentos (append) ---
def log_experiment(row: dict):
    exp_path = os.path.join(os.path.dirname(__file__), "experiments.csv")
    write_header = not os.path.exists(exp_path)
    with open(exp_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
