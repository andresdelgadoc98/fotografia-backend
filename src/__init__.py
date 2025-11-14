from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os

db = SQLAlchemy()

def create_app():
    load_dotenv()

    app = Flask(__name__)
    CORS(app)

    # --- Construcción dinámica del string de conexión ---
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", 5432)
    DB_NAME = os.getenv("DB_NAME")

    DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # --- Registro de blueprints ---
    from .routes import photos,folder,category
    app.register_blueprint(photos.main, url_prefix='/api/photos')
    app.register_blueprint(folder.main, url_prefix='/api/folder')
    app.register_blueprint(category.main, url_prefix='/api/categories')


    return app
