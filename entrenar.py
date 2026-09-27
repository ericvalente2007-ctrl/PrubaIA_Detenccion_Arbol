from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
RUTA_DATA = BASE_DIR / "dataset" / "data.yaml"


def iniciar_entrenamiento():
  modelo = YOLO("yolov8n.pt")

  modelo.train(
      data=str(RUTA_DATA),
      epochs=35,
      imgsz=640,
      batch=4,
      name="Pr1",
  )

  print("\n¡Culmine")


if __name__ == "__main__":
  iniciar_entrenamiento()