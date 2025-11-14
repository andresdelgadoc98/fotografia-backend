from flask import Flask, request, jsonify,Blueprint,send_file, abort,url_for
from pathlib import Path
import time, hashlib, os
from src.utils.utils import run_classification_thread,get_or_create_folder_with_images,partition_diffs,bulk_insert_images,bulk_update_images,mark_deleted,index_embedding,search_in_embeddings

import threading
from src.database.models import Folder, Image
app = Flask(__name__)

main = Blueprint('folder', __name__)

@main.route('/get_process', methods=['GET'])
def get_progress():

    # Buscar folder en proceso
    folder = Folder.query.filter_by(status="processing").first()

    if not folder:
        return {
            "processing": False,
            "message": "No hay ningún folder en procesamiento",
            "total": 0,
            "indexed": 0,
            "pending": 0,
            "percent": 0
        }

    folder_id = folder.id

    total = Image.query.filter_by(folder_id=folder_id).count()
    indexed = Image.query.filter_by(folder_id=folder_id, status="indexed").count()
    pending = total - indexed

    percent = round((indexed / total) * 100, 2) if total > 0 else 0

    return {
        "processing": True,
        "folder_id": folder_id,
        "total": total,
        "indexed": indexed,
        "pending": pending,
        "percent": percent,
    }



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