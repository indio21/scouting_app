"""Entrenamiento del modelo de predicción de potencial.

Este script carga los datos de jugadores desde la base de datos SQLite,
prepara las variables de entrada y las etiquetas, entrena una red
neuronal simple en PyTorch y guarda el modelo entrenado.  También
imprime la precisión en el conjunto de prueba.

Uso:
    python train_model.py --db-url sqlite:///players.db --model-out model.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Player

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


def load_data(db_url: str):
    """Carga los jugadores desde la base de datos y devuelve matrices X, y."""
    engine = create_engine(db_url)
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
        X.append(features)
        y.append(1 if p.potential_label else 0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normaliza las características entre 0 y 1."""
    X_norm = X.copy()
    # Normalizar edad (columna 0) entre 16 y 22
    ages = X[:, 0]
    X_norm[:, 0] = (ages - 16) / (22 - 16)
    # Normalizar habilidades (columna 1..10) dividiendo entre 20
    X_norm[:, 1:] = X_norm[:, 1:] / 20.0
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
    parser.add_argument("--db-url", type=str, default="sqlite:///players.db", help="URL de la base de datos")
    parser.add_argument("--model-out", type=str, default="model.pt", help="Ruta de salida del modelo")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje")
    args = parser.parse_args()
    main(args.db_url, args.model_out, args.epochs, args.lr)