from flask import Flask, request, jsonify,Blueprint,send_file, abort,url_for
from pathlib import Path
import time, hashlib, os
from src.utils.utils import run_classification_thread,get_or_create_folder_with_images,partition_diffs,bulk_insert_images,bulk_update_images,mark_deleted,index_embedding,search_in_embeddings
from src.database.models import Image,ImageAnalysis
import threading

app = Flask(__name__)
base_url = "http://192.168.0.22:5001"
main = Blueprint('documents', __name__)

DB_PATH = "photo_index.db"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic"}


@main.route('scan', methods=['POST'])
def scan_folder():
    data = request.get_json(silent=True) or {}
    path = data.get("path")
    compute_hash = bool(data.get("compute_hash", False))
    allowed_exts = data.get("allowed_exts", [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic"])
    allowed_exts = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_exts]

    if not path:
        return jsonify({"error": "Falta el campo 'path'"}), 400

    base_path = Path(path)
    if not base_path.exists() or not base_path.is_dir():
        return jsonify({"error": "Ruta no válida o no es un directorio"}), 400

    if not os.access(base_path, os.R_OK):
        return jsonify({"error": "Sin permisos de lectura en la carpeta"}), 400

    root_path = str(base_path.resolve())


    folder_id, existing_images = get_or_create_folder_with_images(root_path)

    found_files = []

    for f in base_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in allowed_exts:
            continue

        try:
            stat = f.stat()
            file_info = {
                "abs_path": str(f.resolve()),
                "rel_path": str(f.relative_to(base_path)),
                "size_bytes": stat.st_size,
                "mtime": int(stat.st_mtime)
            }
            found_files.append(file_info)
        except Exception as e:
            print(f"[WARN] No se pudo leer {f}: {e}")
            continue

        # --- Paso 5: Detección de diferencias ---
        existing_paths = set(existing_images.keys())
        found_paths = {f["abs_path"] for f in found_files}

        inserted, updated, unchanged, deleted = 0, 0, 0, 0


    for file_info in found_files:
        abs_path = file_info["abs_path"]
        size_bytes = file_info["size_bytes"]
        mtime = file_info["mtime"]

        if abs_path not in existing_images:
            inserted += 1

        else:
            old = existing_images[abs_path]
            if size_bytes != old["size"] or mtime != old["mtime"]:
                updated += 1

            else:
                unchanged += 1

    for old_path in existing_paths - found_paths:
        deleted += 1


    to_insert, to_update, to_mark_deleted, stats = partition_diffs(found_files, existing_images)


    ins_n = bulk_insert_images(folder_id, to_insert)
    upd_n = bulk_update_images( to_update)
    del_n = mark_deleted(folder_id, to_mark_deleted)

    stats = {"inserted": ins_n, "updated": upd_n, "deleted": del_n}

    return jsonify({
        "path": root_path,
        "folder_id": folder_id,
        "found_count": len(found_files),
        "existing_count": len(existing_images),
        "stats": stats
    })
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



@main.route("view/<int:image_id>", methods=["GET"])
def view_image(image_id):
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


@main.route("/", methods=["GET"])
def list_photos():
    """Devuelve todas las imágenes clasificadas (status='indexed') con su análisis"""
    images = (
        Image.query
        .filter_by(status="indexed")
        .join(ImageAnalysis)
        .order_by(Image.id)
        .all()
    )

    photos = []
    for img in images:
        analysis = img.analysis
        if not analysis:
            continue

        photo_data = {
            "id": img.id,
            "category": analysis.category or "Sin categoría",
            "tags": ", ".join(analysis.tags) if isinstance(analysis.tags, list) else str(analysis.tags or ""),
            "img": f"{base_url}{url_for('documents.view_image', image_id=img.id)}"
        }
        photos.append(photo_data)

    return jsonify(photos)


@main.route('/search_images', methods=['POST'])
def search_images():
    """
    Búsqueda semántica entre las descripciones de imágenes.
    Usa FAISS local generado en db/photos_descriptions
    """
    try:
        data = request.json
        user_search = data.get('user_search', '').strip()

        if not user_search:
            return jsonify({"error": "Campo 'user_search' requerido"}), 400

        print(f"🔍 Consulta recibida: {user_search}")

        return search_in_embeddings(user_search)
    except Exception as e:
        print(f"⚠️ Error en búsqueda semántica: {e}")
        return jsonify({"error": str(e)}), 500
