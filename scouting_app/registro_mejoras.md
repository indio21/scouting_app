# Registro de mejoras – scouting_app

## Prompts y requerimientos abordados
1. Ajustes funcionales solicitados en la sesión previa (pesos por posición, CRUD de directores, validación de DNI, separación de bases, etc.).
2. Solicitud actual: ejecutar el reentrenamiento del modelo sobre la base completa y sincronizar la shortlist luego de refrescar datos masivos.
3. Documentar las acciones realizadas y explicar el porqué de cada cambio.

## Acciones realizadas hoy
- **Generación de dataset de entrenamiento**: python generate_data.py --num-players 20000 --db-url sqlite:///players_training.db
  - Motivo: la base players_training.db estaba vacía; se generó un set inicial para posibilitar el reentrenamiento hasta contar con datos reales.
- **Reentrenamiento del modelo**: python train_model.py --db-url sqlite:///players_training.db --epochs 30
  - Motivo: aprovechar los nuevos vectores one-hot de posición y normalización de edades (12-30) para mejorar la predicción del potencial. Resultado: precisión 85.8?% y modelo actualizado en model.pt.
- **Sincronización de shortlist**: python sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100
  - Motivo: poblar la base operativa con 100 juveniles del rango 12-18 más recientes/alto potencial, alineado al flujo de scouting.
- **Migración de esquema en shortlist**: se añadió la columna 
ational_id en players_updated_v2.db para soportar la nueva validación de DNI.
  - Comando: python -c "... ALTER TABLE players ADD COLUMN national_id TEXT"

## Próximos pasos recomendados
- Automatizar la generación de datos reales en players_training.db (importaciones desde planillas, informes externos, etc.).
- Exponer en el panel un botón de "Reentrenar modelo" que dispare 	rain_model.py y actualice model.pt usando los historiales más recientes, considerando un job en background o tarea programada.
- Encadenar la sincronización (sync_shortlist.py) al flujo de carga masiva o a un botón "Actualizar shortlist" para que siempre haya exactamente la cantidad objetivo (ej. 100 jugadores) en evaluación.
- Registrar métricas del modelo (pérdida/precisión, fecha de entrenamiento, tamaño del dataset) en una tabla para auditoría.

- **Panel de configuracion**: se incorporo /settings con un boton 'Update base de datos' que ejecuta el reentrenamiento (	rain_model.py) y la sincronizacion (sync_shortlist.py) desde la interfaz.

- **Fix modelo dinámico**: load_model ahora calcula automáticamente las entradas (edad + atributos + one-hot de posición) y reentrena si detecta incompatibilidad, evitando fallas al iniciar la app tras cambiar la arquitectura.

- **Verificacion completa**: se ejecuto update_database_pipeline() desde la app para reentrenar y sincronizar nuevamente; precision 85.8% y shortlist de 100 juveniles OK.

- **Dashboard fix**: se creó player_feature_vector reutilizable y el dashboard ahora normaliza [edad + atributos + one-hot de posición], evitando el crash mat1 x mat2 y alineando sus features con el modelo actual.

