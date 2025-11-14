from flask import Blueprint, request, jsonify
from src.database.models import Category, ImageAnalysis, db

main = Blueprint('category', __name__)

@main.route("/", methods=["GET"])
def get_categories():
    q = request.args.get("q", "").strip().lower()
    query = Category.query
    if q:
        query = query.filter(Category.name.ilike(f"%{q}%"))
    categories = query.all()
    results = []
    for c in categories:
        count = ImageAnalysis.query.filter_by(category_id=c.id).count()
        results.append({
            "id": c.id,
            "name": c.name,
            "count": count
        })

    return jsonify({
        "total": len(results),
        "categories": results
    })
