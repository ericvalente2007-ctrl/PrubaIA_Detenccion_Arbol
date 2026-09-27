from pathlib import Path
from tkinter import filedialog
from PIL import Image
import customtkinter as ctk
from ultralytics import YOLO

ctk.set_appearance_mode("Light")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "Pr1" / "weights" / "best.pt"


class ValidadorDetectorIA(ctk.CTk):

  def __init__(self):
    super().__init__()
    self.title("Prueba IA - Detector de Árboles")
    self.geometry("600x500")
    self.configure(fg_color="white")

    self.modelo = YOLO(str(MODEL_PATH))

    self.btn_cargar = ctk.CTkButton(
        self,
        text="Subir imagen",
        command=self.procesar_y_mostrar,
        fg_color="white",
        text_color="black",
        hover_color="#E0E0E0",
        border_color="#CCCCCC",
        border_width=1,
    )
    self.btn_cargar.pack(pady=10)

    self.lbl_estado = ctk.CTkLabel(self, text="", text_color="black")
    self.lbl_estado.pack(pady=5)

    self.lbl_imagen = ctk.CTkLabel(self, text="")
    self.lbl_imagen.pack(pady=10)

  def procesar_y_mostrar(self):
    filepath = filedialog.askopenfilename(
        filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.tif")]
    )
    if not filepath:
      return

    results = self.modelo.predict(source=filepath, conf=0.10, save=False)

    res_array = results[0].plot()
    img_resultado = Image.fromarray(res_array)
    img_resultado.thumbnail((500, 400))
    ctk_img = ctk.CTkImage(
        light_image=img_resultado,
        dark_image=img_resultado,
        size=img_resultado.size,
    )

    self.lbl_imagen.configure(image=ctk_img)
    self.lbl_estado.configure(
        text=f"Detecciones encontradas: {len(results[0].boxes)}"
    )


if __name__ == "__main__":
  app = ValidadorDetectorIA()
  app.mainloop()