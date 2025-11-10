from datetime import datetime
from src import db


class Folder(db.Model):
    __tablename__ = "folders"

    id = db.Column(db.Integer, primary_key=True)
    root_path = db.Column(db.Text, unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relación 1:N con imágenes
    images = db.relationship("Image", back_populates="folder")


class Image(db.Model):
    __tablename__ = "images"

    id = db.Column(db.Integer, primary_key=True)
    folder_id = db.Column(db.Integer, db.ForeignKey("folders.id"))
    abs_path = db.Column(db.Text, unique=True, nullable=False)
    rel_path = db.Column(db.Text)
    size_bytes = db.Column(db.BigInteger)
    mtime = db.Column(db.BigInteger)
    sha256 = db.Column(db.Text)
    status = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    folder = db.relationship("Folder", back_populates="images")
    analysis = db.relationship("ImageAnalysis", back_populates="image", uselist=False)


class ImageAnalysis(db.Model):
    __tablename__ = "image_analysis"

    image_id = db.Column(db.Integer, db.ForeignKey("images.id"), primary_key=True)
    description = db.Column(db.Text)
    category = db.Column(db.String(100))
    subcategory = db.Column(db.String(100))
    tags = db.Column(db.JSON)
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)

    image = db.relationship("Image", back_populates="analysis")