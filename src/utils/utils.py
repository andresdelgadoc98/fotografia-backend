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
from src.database.models import Image,ImageAnalysis,Category
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.ollama_client import analyze_image_with_ollama
from sqlalchemy import func
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

def search_in_embeddings(
    user_search,
    start_date=None,
    end_date=None,
    category=None,
    limit=50,
    page=1,
    embedding_model="nomic-embed-text",
    base_url="http://192.168.0.21:11434",
    k=50,   # aumentamos FAISS para tener margen para filtrar
):
    try:
        # --- Validación ---
        if not user_search.strip():
            raise ValueError("Texto vacío en user_search")

        db_path = "db/photos_descriptions"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No existe índice FAISS en {db_path}")

        # --- Embeddings client ---
        embedding = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url
        )

        # --- Load FAISS ---
        vector_store = FAISS.load_local(
            db_path,
            embedding,
            allow_dangerous_deserialization=True
        )

        # --- FAISS similarity search ---
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(user_search)

        image_ids_ranked = [r.metadata.get("id") for r in results if r.metadata.get("id")]

        if not image_ids_ranked:
            return jsonify({"results": [], "total": 0})

        # --- SQL query base ---
        query = (
            Image.query
            .join(Image.analysis)
            .filter(Image.id.in_(image_ids_ranked))
        )


        # --- Filtro categoría ---
        if category:
            clean = category.strip().lower()
            query = query.join(ImageAnalysis.category_rel).filter(
                func.lower(Category.name) == clean
            )

        # --- Filtro fechas ---
        if start_date:
            try:
                dt_start = datetime.fromisoformat(start_date)
                query = query.filter(ImageAnalysis.analyzed_at >= dt_start)
            except:
                pass

        if end_date:
            try:
                dt_end = datetime.fromisoformat(end_date)
                query = query.filter(ImageAnalysis.analyzed_at <= dt_end)
            except:
                pass

        # --- Obtener resultados filtrados ---
        images = query.all()

        # --- Mantener orden por FAISS ---
        images_sorted = sorted(images, key=lambda img: image_ids_ranked.index(img.id))

        # --- Paginación ---
        start_i = (page - 1) * limit
        end_i = start_i + limit

        page_items = images_sorted[start_i:end_i]
        total = len(images_sorted)

        # --- Serializar ---
        output = []
        for img in page_items:
            ia = img.analysis
            output.append({
                "id": img.id,
                "abs_path": img.abs_path,
                "description": ia.description if ia else None,
                "category": ia.category_rel.name if ia and ia.category_rel else None,
                "subcategory": ia.subcategory if ia else None,
                "tags": ia.tags if ia else None,
                "analyzed_at": ia.analyzed_at.isoformat() if ia and ia.analyzed_at else None,
                "img": url_for("documents.view_image", image_id=img.id, _external=True)
            })

        return jsonify({
            "results": output,
            "total": total,
            "page": page,
            "limit": limit,
        })

    except Exception as e:
        print(f"⚠️ Error en search_in_embeddings: {e}")
        raise


def process_single_image(img, model, base_prompt, retry_non_json):
    """Procesa una sola imagen y devuelve resultado resumido."""

    try:
        result = analyze_image_with_ollama(
            image_path=img.abs_path,
            model=model,
            prompt=base_prompt,
            retry_non_json=retry_non_json
        )

        if "error" in result:
            return {"id": img.id, "status": "error", "error": result["error"]}

        desc = result.get("description", "").strip()
        cat = (result.get("category") or "Sin clasificar").strip().title()
        sub = result.get("subcategory") or None
        tags = result.get("tags") or []

        index_embedding(img.id, desc, cat, embedding_model="nomic-embed-text")

        # Devolver resultado limpio
        return {
            "id": img.id,
            "status": "ok",
            "desc": desc,
            "category": cat,
            "subcategory": sub,
            "tags": tags,

        }

    except Exception as e:
        return {"id": img.id, "status": "error", "error": str(e)}


def get_or_create_category(name):
    if not name:
        return None

    clean = name.strip().lower()

    cat = Category.query.filter(func.lower(Category.name) == clean).first()
    if cat:
        return cat.id


    cat = Category(name=clean)
    db.session.add(cat)
    db.session.commit()
    return cat.id

def run_classification_thread(app, pending_images, folder_id, model, retry_non_json):
    with app.app_context():
        print(f"🚀 Clasificación en background — {len(pending_images)} imágenes (4 hilos)")
        folder = Folder.query.get(folder_id)
        if folder:
            folder.status = "processing"
            db.session.commit()

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

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_single_image, img, model, base_prompt, retry_non_json)
                for img in pending_images
            ]

            for future in as_completed(futures):
                result = future.result()

                if result["status"] == "ok":
                    img = Image.query.get(result["id"])
                    if not img:
                        continue

                    img.status = "indexed"
                    img.updated_at = datetime.utcnow()

                    # NUEVO -> normalizar categoría
                    cat_id = get_or_create_category(result["category"])

                    analysis = ImageAnalysis.query.filter_by(image_id=img.id).first()

                    if not analysis:
                        analysis = ImageAnalysis(
                            image_id=img.id,
                            description=result["desc"],
                            category_id=cat_id,
                            subcategory=result["subcategory"],
                            tags=result["tags"],
                            analyzed_at=datetime.utcnow(),
                        )
                        db.session.add(analysis)
                    else:
                        analysis.description = result["desc"]
                        analysis.category_id = cat_id
                        analysis.subcategory = result["subcategory"]
                        analysis.tags = result["tags"]
                        analysis.analyzed_at = datetime.utcnow()

                    db.session.commit()

            print(f"🏁 Clasificación terminada para folder {folder_id}")

def search_with_filters(start_date, end_date, category, limit=50, page=1):
    from sqlalchemy import func
    query = Image.query.join(Image.analysis)

    if start_date:
        query = query.filter(ImageAnalysis.analyzed_at >= start_date)

    if end_date:
        query = query.filter(ImageAnalysis.analyzed_at <= end_date)

    if category:
        clean = category.strip().lower()
        query = query.join(ImageAnalysis.category_rel).filter(
            func.lower(Category.name) == clean
        )

    total = query.count()

    images = (
        query
        .order_by(Image.id.asc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    results = []
    for img in images:
        ia = img.analysis
        results.append({
            "id": img.id,
            "img": url_for("documents.view_image", image_id=img.id, _external=True),
            "category": ia.category_rel.name if ia and ia.category_rel else None,
            "description": ia.description if ia else None,
            "tags": ia.tags if ia else None,
        })

    return jsonify({
        "results": results,
        "total": total,
        "page": page,
        "limit": limit
    })


def list_all_photos(limit=50, page=1):
    query = Image.query.join(Image.analysis)
    total = query.count()

    images = (
        query
        .order_by(Image.id.asc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    results = []
    for img in images:
        ia = img.analysis
        results.append({
            "id": img.id,
            "img": url_for("documents.view_image", image_id=img.id, _external=True),
            "category": ia.category_rel.name if ia and ia.category_rel else None,
            "description": ia.description if ia else None,
            "tags": ia.tags if ia else None,
        })

    return jsonify({
        "results": results,
        "total": total,
        "page": page,
        "limit": limit
    })
