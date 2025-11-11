from datetime import datetime
from src.database.models import Folder, Image
from src import db
import os
import faiss
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from flask import jsonify,url_for
from src.database.models import Image,ImageAnalysis
from src.utils.ollama_client import analyze_image_with_ollama
def ensure_folder_record(root_path: str):
    folder = Folder.query.filter_by(root_path=root_path).first()
    if folder:
        return folder.id

    folder = Folder(root_path=root_path, created_at=datetime.utcnow())
    db.session.add(folder)
    db.session.commit()
    return folder.id


# ---------------------------------------------------------------------
# Compara disco vs base (igual que antes, sin SQL)
# ---------------------------------------------------------------------
def partition_diffs(found_files, existing_images):
    existing_paths = set(existing_images.keys())
    found_paths = {f["abs_path"] for f in found_files}

    to_insert = []
    to_update = []
    to_mark_deleted = list(existing_paths - found_paths)

    for f in found_files:
        ap = f["abs_path"]
        if ap not in existing_images:
            to_insert.append(f)
        else:
            old = existing_images[ap]
            if f["size_bytes"] != old["size"] or f["mtime"] != old["mtime"]:
                to_update.append(f)

    stats = {
        "inserted": len(to_insert),
        "updated": len(to_update),
        "deleted": len(to_mark_deleted)
    }
    return to_insert, to_update, to_mark_deleted, stats


# ---------------------------------------------------------------------
# Inserta nuevas imágenes
# ---------------------------------------------------------------------
def bulk_insert_images(folder_id, rows):
    if not rows:
        return 0

    now = datetime.utcnow()
    images = [
        Image(
            folder_id=folder_id,
            abs_path=r["abs_path"],
            rel_path=r["rel_path"],
            size_bytes=r["size_bytes"],
            mtime=r["mtime"],
            sha256=r.get("sha256"),
            status="new",
            created_at=now,
            updated_at=now
        )
        for r in rows
    ]
    db.session.bulk_save_objects(images)
    db.session.commit()
    return len(images)


# ---------------------------------------------------------------------
# Actualiza modificados
# ---------------------------------------------------------------------
def bulk_update_images(rows):
    if not rows:
        return 0

    updated_count = 0
    now = datetime.utcnow()

    for r in rows:
        image = Image.query.filter_by(abs_path=r["abs_path"]).first()
        if image:
            image.size_bytes = r["size_bytes"]
            image.mtime = r["mtime"]
            image.sha256 = r.get("sha256")
            image.status = "new"
            image.updated_at = now
            updated_count += 1

    db.session.commit()
    return updated_count


# ---------------------------------------------------------------------
# Marca como eliminados
# ---------------------------------------------------------------------
def mark_deleted(folder_id, abs_paths):
    if not abs_paths:
        return 0

    images = Image.query.filter(
        Image.folder_id == folder_id,
        Image.abs_path.in_(abs_paths)
    ).all()

    now = datetime.utcnow()
    for img in images:
        img.status = "deleted"
        img.updated_at = now

    db.session.commit()
    return len(images)


def get_or_create_folder_with_images(root_path: str):
    """
    Asegura que exista un folder con ese root_path y obtiene sus imágenes registradas.

    Retorna:
        folder_id (int)
        existing_images (dict)
    """
    # Buscar el folder
    folder = Folder.query.filter_by(root_path=root_path).first()

    # Si no existe, crearlo
    if not folder:
        folder = Folder(root_path=root_path, created_at=datetime.utcnow())
        db.session.add(folder)
        db.session.commit()

    # Obtener las imágenes existentes
    images = Image.query.filter_by(folder_id=folder.id).all()

    # Convertir a diccionario {abs_path: {...}}
    existing_images = {
        img.abs_path: {
            "size": img.size_bytes,
            "mtime": img.mtime,
            "sha256": img.sha256
        }
        for img in images
    }

    return folder.id, existing_images

def index_embedding(image_id, description, category,embedding_model):
    """
    Genera y guarda el embedding de una imagen analizada
    usando FAISS de forma local.
    """
    try:
        # Ruta donde guardas tus índices FAISS
        db_path = "db/photos_descriptions"

        embedding = OllamaEmbeddings(model=embedding_model, base_url="http://192.168.0.21:11434",)

        # --- Carga o creación del índice ---
        if not os.path.exists(db_path):
            index = faiss.IndexFlatL2(len(embedding.embed_query("init")))
            vector_store = FAISS(
                embedding_function=embedding,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        else:
            vector_store = FAISS.load_local(
                db_path, embedding, allow_dangerous_deserialization=True
            )

        # --- Crea el documento ---
        document = Document(
            page_content=description,
            metadata={"id": image_id, "category": category},
        )

        # --- Inserta y guarda ---
        vector_store.add_documents(documents=[document], ids=[str(image_id)])
        vector_store.save_local(db_path)

        print(f"✔ Embedding guardado para imagen {image_id}")

    except Exception as e:
        print(f"⚠️ Error al indexar imagen {image_id}: {e}")

def search_in_embeddings(user_search, embedding_model="nomic-embed-text", base_url="http://192.168.0.21:11434", k=10):
    """
    Realiza una búsqueda semántica en el índice FAISS de descripciones de imágenes
    y devuelve los resultados con el mismo formato que list_photos().
    """
    try:
        # --- Validación ---
        if not user_search or not user_search.strip():
            raise ValueError("El parámetro 'user_search' está vacío.")

        db_path = "db/photos_descriptions"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No existe el índice FAISS en {db_path}")

        # --- Crear cliente de embeddings ---
        embedding = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url
        )

        # --- Cargar índice FAISS ---
        vector_store = FAISS.load_local(
            db_path,
            embedding,
            allow_dangerous_deserialization=True
        )

        # --- Buscar similitud ---
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(user_search)

        # --- Obtener IDs de imagen ---
        image_ids = [r.metadata.get("id") for r in results if r.metadata.get("id")]

        if not image_ids:
            print("⚠️ Sin resultados FAISS.")
            return jsonify([])

        # --- Consultar base de datos ---
        images = (
            Image.query
            .filter(Image.id.in_(image_ids))
            .join(ImageAnalysis)
            .all()
        )

        # --- Formato frontend (igual que list_photos) ---
        photos = []
        for img in images:
            analysis = img.analysis
            if not analysis:
                continue
            photos.append({
                "id": img.id,
                "category": analysis.category or "Sin categoría",
                "tags": ", ".join(analysis.tags) if isinstance(analysis.tags, list) else str(analysis.tags or ""),
                "img": url_for("documents.view_image", image_id=img.id, _external=True)
            })

        print(f"✅ {len(photos)} resultados encontrados.")
        return jsonify(photos)

    except Exception as e:
        print(f"⚠️ Error en search_in_embeddings: {e}")
        raise


def run_classification_thread(app, pending_images, folder_id, model, retry_non_json):
    """
    Procesa las imágenes en background dentro del contexto Flask.
    """
    with app.app_context():  # 🔥 Activa el contexto de Flask aquí
        print(f"🚀 Clasificación iniciada en background ({len(pending_images)} imágenes)")

        base_prompt = (
            "Describe brevemente la imagen en español y devuelve SOLO JSON con:\n"
            "{\n"
            '  \"description\": str,\n'
            '  \"category\": str,\n'
            '  \"subcategory\": str|null,\n'
            '  \"tags\": [str]\n'
            "}\n"
            "No incluyas texto fuera del JSON. Sé conciso."
        )

        analyzed, errors = [], []

        for img in pending_images:
            try:
                result = analyze_image_with_ollama(
                    image_path=img.abs_path,
                    model=model,
                    prompt=base_prompt,
                    retry_non_json=retry_non_json
                )

                if "error" in result:
                    img.status = "error"
                    img.updated_at = datetime.utcnow()
                    errors.append({"image_id": img.id, "error": result["error"]})
                    continue

                desc = result.get("description", "").strip()
                cat = (result.get("category") or "Sin clasificar").strip().title()
                sub = result.get("subcategory") or None
                tags = result.get("tags") or []

                index_embedding(
                    image_id=img.id,
                    description=desc,
                    category=cat,
                    embedding_model="nomic-embed-text"
                )

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

                img.status = "indexed"
                img.updated_at = datetime.utcnow()
                analyzed.append(img.id)

            except Exception as e:
                print(f"⚠️ Error procesando {img.id}: {e}")
                errors.append({"image_id": img.id, "error": str(e)})

        db.session.commit()
        print(f"✅ Clasificación completada — {len(analyzed)} procesadas, {len(errors)} con error.")
