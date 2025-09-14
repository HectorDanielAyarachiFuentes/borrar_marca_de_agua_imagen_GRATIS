from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
import uuid
from PIL import Image

app = Flask(__name__)

# --- Configuración ---
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["SECRET_KEY"] = "supersecretkey_cambiame"  # Necesario para mensajes flash

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Carga de Modelos (Avanzado) ---
# Se intentará cargar los modelos de IA. Esto puede tardar y consumir mucha RAM.
STABLE_DIFFUSION_AVAILABLE = False
sd_pipe = None
try:
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    print("INFO: Cargando modelo de Stable Diffusion... Esto puede tardar varios minutos.")
    device = "cpu"  # Forzamos el uso de CPU para evitar errores de memoria en la GPU.
    print(f"INFO: Usando dispositivo: {device} para Stable Diffusion.")
    
    # Usar float16 para ahorrar memoria en GPU, de lo contrario float32 en CPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        dtype=dtype,
    ).to(device)

    # Optimización: Mueve partes del modelo a la CPU para ahorrar memoria VRAM.
    # Esto es crucial para GPUs con poca memoria (como 2GB o 4GB).
    # Solo se aplica si estamos usando la GPU.
    if device == "cuda":
        sd_pipe.enable_sequential_cpu_offload()

    print("INFO: Modelo de Stable Diffusion cargado exitosamente.")
    STABLE_DIFFUSION_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: 'torch' o 'diffusers' no están instalados. La opción de Stable Diffusion no estará disponible.")
    print("Para habilitarla, ejecuta: pip install torch diffusers transformers accelerate")
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo de Stable Diffusion: {e}")
    print("ADVERTENCIA: La opción de Stable Diffusion no estará disponible.")


def allowed_file(filename):
    """Verifica si la extensión del archivo es permitida."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No se encontró el archivo en la solicitud.")
        return redirect(url_for("index"))
    
    file = request.files["file"]
    if file.filename == "":
        flash("No se seleccionó ningún archivo.")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        # Redirigir a la página de edición en lugar de procesar aquí
        return redirect(url_for("edit_image", filename=unique_filename))
    else:
        flash("Tipo de archivo no permitido. Sube archivos de imagen (png, jpg, jpeg, gif).")
        return redirect(url_for("index"))


@app.route("/edit/<filename>")
def edit_image(filename):
    """Muestra la página de edición con la imagen subida."""
    # Pasamos la bandera de disponibilidad del modelo a la plantilla
    return render_template("edit.html", filename=filename, stablediffusion_available=STABLE_DIFFUSION_AVAILABLE)


@app.route("/process", methods=["POST"])
def process_image():
    """Recibe la máscara dibujada y procesa la imagen."""
    try:
        filename = request.form.get("filename")
        mask_data_url = request.form.get("maskData")
        algorithm = request.form.get("algorithm", "opencv") # Valor por defecto

        if not filename or not mask_data_url:
            flash("Faltan datos para el procesamiento (nombre de archivo o máscara).")
            return redirect(url_for("index"))

        # --- Decodificar la máscara desde Base64 ---
        # 1. Separar el prefijo 'data:image/png;base64,'
        header, encoded = mask_data_url.split(",", 1)
        
        # 2. Decodificar la data
        mask_bytes = base64.b64decode(encoded)
        
        # 3. Convertir los bytes a un array de numpy y luego a una imagen de OpenCV
        nparr = np.frombuffer(mask_bytes, np.uint8)
        # Usamos IMREAD_UNCHANGED para conservar el canal alfa (la transparencia)
        decoded_mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # --- Preparar la máscara para inpainting ---
        # La máscara para inpaint debe ser de 1 canal (escala de grises).
        # El dibujo está en el canal alfa (el cuarto canal de un PNG).
        if decoded_mask.shape[2] == 4:
            mask = decoded_mask[:, :, 3]
        else:
            # Si por alguna razón no tiene canal alfa, la convertimos a escala de grises
            mask = cv2.cvtColor(decoded_mask, cv2.COLOR_BGR2GRAY)

        # --- Cargar imagen original y escalar la máscara ---
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("No se pudo leer la imagen original desde el servidor.")

        original_height, original_width, _ = img.shape

        # Redimensionar la máscara para que coincida con el tamaño de la imagen original
        resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Asegurarse de que la máscara sea binaria (0 o 255)
        _, final_mask = cv2.threshold(resized_mask, 1, 255, cv2.THRESH_BINARY)

        # --- Aplicar el algoritmo de inpainting seleccionado ---
        if algorithm == "opencv":
            print("INFO: Procesando con OpenCV (INPAINT_TELEA)")
            result = cv2.inpaint(img, final_mask, 3, cv2.INPAINT_TELEA)
        
        elif algorithm == "scikit-image":
            print("INFO: Procesando con Scikit-image (inpaint_biharmonic)")
            from skimage.restoration import inpaint_biharmonic
            # Scikit-image trabaja con imágenes RGB en formato flotante (0 a 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask_bool = final_mask > 0
            
            result_float = inpaint_biharmonic(img_rgb, mask_bool, channel_axis=-1)
            
            # Convertir de vuelta a formato de OpenCV (BGR, 0 a 255)
            result_uint8_rgb = (result_float * 255).astype(np.uint8)
            result = cv2.cvtColor(result_uint8_rgb, cv2.COLOR_RGB2BGR)

        elif algorithm == "stablediffusion" and STABLE_DIFFUSION_AVAILABLE:
            print("INFO: Procesando con Stable Diffusion")
            # Convertir imágenes de OpenCV (BGR) a PIL (RGB)
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(final_mask)

            # Guardar las dimensiones originales para restaurarlas después.
            # Esto es crucial porque el modelo de IA trabaja con imágenes cuadradas (p.ej. 512x512)
            # y estira la imagen de entrada si no es cuadrada, causando deformaciones.
            original_width, original_height = pil_img.size

            # Un 'prompt' genérico para guiar a la IA
            prompt = "high quality photography, sharp focus, detailed"

            # Ejecutar el pipeline de IA. La imagen resultante será cuadrada.
            result_pil_square = sd_pipe(prompt=prompt, image=pil_img, mask_image=pil_mask).images[0]

            # Redimensionar la imagen resultante (que es cuadrada) de vuelta a las dimensiones originales.
            # Esto corrige el problema de la imagen estirada.
            result_pil = result_pil_square.resize((original_width, original_height), Image.Resampling.LANCZOS)

            # Convertir la imagen resultante de PIL de vuelta a formato OpenCV
            result_rgb = np.array(result_pil)
            result = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        else:
            flash(f"Algoritmo '{algorithm}' no es válido o no está disponible.")
            return redirect(url_for("edit_image", filename=filename))

        # --- Guardar y mostrar el resultado ---
        result_filename = "result_" + filename
        result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
        cv2.imwrite(result_path, result)

        return render_template("result.html", original=filename, result=result_filename)
    except Exception as e:
        flash(f"Ocurrió un error durante el procesamiento: {e}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
