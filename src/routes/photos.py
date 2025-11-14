from flask import Flask, request, jsonify,Blueprint,send_file, abort,url_for
from pathlib import Path
import time, hashlib, os
from src.utils.utils import run_classification_thread,search_in_embeddings,search_with_filters,list_all_photos
from src.database.models import Image,ImageAnalysis
import threading
from PIL import Image as PILImage
from io import BytesIO
app = Flask(__name__)
base_url = "http://192.168.0.21:5001"
main = Blueprint('documents', __name__)

DB_PATH = "photo_index.db"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic"}


from flask import current_app

@main.route("/classify", methods=["POST"])
def classify_images():
    data = request.get_json(silent=True) or {}
    folder_id = data.get("folder_id")
    limit = int(data.get("limit", 1000))
    model = data.get("model", "llava")
    ids = data.get("ids")
    retry_non_json = bool(data.get("retry_non_json", True))

    if not folder_id:
        return jsonify({"error": "Falta el campo 'folder_id'"}), 400

    # Obtener imágenes pendientes
    if ids:
        pending_images = Image.query.filter(
            Image.id.in_(ids), Image.status == "new"
        ).all()
    else:
        pending_images = (
            Image.query.filter_by(folder_id=folder_id, status="new")
            .order_by(Image.id)
            .limit(limit)
            .all()
        )

    if not pending_images:
        return jsonify({"message": "No hay imágenes nuevas para analizar"}), 200

    # Captura la instancia actual de Flask
    app = current_app._get_current_object()

    # Lanza el hilo pasando la app
    thread = threading.Thread(
        target=run_classification_thread,
        args=(app, pending_images, folder_id, model, retry_non_json),
        daemon=True
    )
    thread.start()

    return jsonify({
        "status": "processing",
        "message": f"Procesando {len(pending_images)} imágenes en background",
        "folder_id": folder_id,
        "model": model
    }), 202



@main.route("download/<int:image_id>", methods=["GET"])
def download_image(image_id):
    """Devuelve la imagen física según su ID"""
    image = Image.query.get(image_id)
    if not image:
        return abort(404, description="Imagen no encontrada")

    if not os.path.exists(image.abs_path):
        return abort(404, description="Archivo físico no encontrado")

    try:

        return send_file(
            image.abs_path,
            mimetype="image/jpeg" if image.abs_path.lower().endswith(".jpg") or image.abs_path.lower().endswith(".jpeg") else "image/png",
            as_attachment=False
        )
    except Exception as e:
        return abort(500, description=f"Error al servir imagen: {e}")


@main.route("view/<int:image_id>", methods=["GET"])
def view_image(image_id):
    image = Image.query.get(image_id)
    if not image:
        return abort(404, "Imagen no encontrada")

    if not os.path.exists(image.abs_path):
        return abort(404, "Archivo físico no encontrado")

    try:
        # --- Abrir imagen ---
        pil_img = PILImage.open(image.abs_path)

        # --- Reducción opcional ---
        # Baja resolución (ej: max 1024px)
        pil_img.thumbnail((1024, 1024))  # ajusta a tu gusto

        # --- Convertir a bytes con menos calidad ---
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=70)  # calidad 0-100, 70 recomendado
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="image/jpeg",
            as_attachment=False
        )

    except Exception as e:
        return abort(500, f"Error al servir imagen: {e}")


@main.route("/search_images", methods=["POST"])
def search_images():
    try:
        data = request.json or {}

        # Buscar texto del usuario (puede ser vacío)
        user_search = (data.get("user_search") or "").strip()

        # Filtros opcionales
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        category = data.get("category")
        limit = int(data.get("limit") or 50)
        page = int(data.get("page") or 1)

        # ❗ Caso 1: NO hay búsqueda, sí hay filtros
        filters_present = any([start_date, end_date, category])
        if user_search == "" and filters_present:
            return search_with_filters(
                start_date=start_date,
                end_date=end_date,
                category=category,
                limit=limit,
                page=page
            )

        # ❗ Caso 2: NO hay búsqueda y NO hay filtros
        if user_search == "" and not filters_present:
            return list_all_photos(limit=limit, page=page)

        # ❗ Caso 3: Hay búsqueda → usar FAISS
        return search_in_embeddings(
            user_search=user_search,
            start_date=start_date,
            end_date=end_date,
            category=category,
            limit=limit,
            page=page,
        )

    except Exception as e:
        print(f"⚠️ Error en /search_images: {e}")
        return jsonify({"error": str(e)}), 500
