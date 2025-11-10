from flask import Flask, request, jsonify,Blueprint,send_file, abort,url_for
from pathlib import Path
import time, hashlib, os
from src.utils.utils import get_or_create_folder_with_images,partition_diffs,bulk_insert_images,bulk_update_images,mark_deleted,index_embedding,search_in_embeddings
from src.database.models import Image,ImageAnalysis
from src.utils.ollama_client import analyze_image_with_ollama
from datetime import datetime
from src import db

app = Flask(__name__)
base_url = "http://192.168.0.22:5001"
main = Blueprint('documents', __name__)

DB_PATH = "photo_index.db"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic"}

def compute_sha256(file_path):

    """Calcula hash SHA-256 en bloques (opcional, puede ser lento)."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def get_folder_id(conn, root_path):
    cur = conn.cursor()
    cur.execute("SELECT id FROM folders WHERE root_path = ?", (root_path,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO folders (root_path, created_at) VALUES (?, ?)", (root_path, time.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    return cur.lastrowid

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

@main.route("classify", methods=["POST"])
def classify_images():

    data = request.get_json(silent=True) or {}
    folder_id = data.get("folder_id")
    limit = int(data.get("limit", 1000))
    model = data.get("model", "llava")
    ids = data.get("ids")
    retry_non_json = bool(data.get("retry_non_json", True))


    if not folder_id:
        return jsonify({"error": "Falta el campo 'folder_id'"}), 400

    # --- Paso 2: obtener imágenes pendientes ---
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

    # --- Paso 3: definir prompt base ---
    base_prompt = (
        "Describe brevemente la imagen en español y devuelve SOLO JSON con:\n"
        "{\n"
        '  "description": str,\n'
        '  "category": str,     \n'
        '  "subcategory": str|null, \n'
        '  "tags": [str]            // 3-8 palabras clave\n'
        "}\n"
        "No incluyas texto fuera del JSON. Sé conciso."
    )

    # --- Paso 4: ejecución del modelo ---
    analyzed, errors = [], []
    for img in pending_images:
        result = analyze_image_with_ollama(
            image_path=img.abs_path,
            model=model,
            prompt=base_prompt,
            retry_non_json=retry_non_json
        )



        # --- Manejo de errores ---
        if "error" in result:
            img.status = "error"
            img.updated_at = datetime.utcnow()
            errors.append({"image_id": img.id, "error": result["error"]})
            continue

        # --- Normalización de campos ---
        desc = result.get("description", "").strip()
        cat = (result.get("category") or "Sin clasificar").strip().title()
        sub = (result.get("subcategory") or None)
        tags = result.get("tags") or []

        index_embedding(image_id=img.id,
                        description=desc,
                        category=cat,
                        embedding_model="nomic-embed-text")

        # --- Guardar o actualizar análisis ---
        analysis = ImageAnalysis.query.filter_by(image_id=img.id).first()
        if not analysis:
            analysis = ImageAnalysis(
                image_id=img.id,
                description=desc,
                category=cat,
                subcategory=sub,
                tags=tags,
                analyzed_at=datetime.utcnow(),
            )
            db.session.add(analysis)
        else:
            analysis.description = desc
            analysis.category = cat
            analysis.subcategory = sub
            analysis.tags = tags
            analysis.analyzed_at = datetime.utcnow()

        # Marcar imagen como procesada
        img.status = "indexed"
        img.updated_at = datetime.utcnow()

        analyzed.append({
            "image_id": img.id,
            "rel_path": img.rel_path,
            "category": cat,
            "subcategory": sub,
            "tags": tags
        })

    # --- Commit general ---
    db.session.commit()

    # --- Respuesta ---
    return jsonify({
        "folder_id": folder_id,
        "model": model,
        "analyzed": len(analyzed),
        "errors": len(errors),
        "samples": analyzed[:5],
        "errors_sample": errors[:3]
    })


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
