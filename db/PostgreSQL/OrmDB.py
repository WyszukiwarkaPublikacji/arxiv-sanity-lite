import sys
import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

from dotenv import load_dotenv, dotenv_values

from sqlalchemy.sql import func
load_dotenv()


class Base(DeclarativeBase):
    pass

# Create the SQLAlchemy instance, specifying the base class
db = SQLAlchemy(model_class=Base)

# Initialize the Flask app
app = Flask(__name__)
# Set the PostgreSQL database URI
app.config['SQLALCHEMY_DATABASE_URI'] = (
        'postgresql://' + os.getenv('DB_USERNAME') + ':' +
        os.getenv('DB_PASSWORD') + '@localhost:5432/' +
        os.getenv('DB_DATABASE')
)

#app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://'+(os.getenv('DB_USERNAME'))+':'+(os.getenv('DB_PASSWORD'))+'@pgsql:5432/'+(os.getenv('DB_DATABASE'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Link the SQLAlchemy instance with the app
db.init_app(app)

# Define your models by inheriting from db.Model
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(30), unique=True, nullable=False)
    last_name = db.Column(db.String(30), unique=True, nullable=False)
    login = db.Column(db.String(30), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    date_birth = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())
    updated_at = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())
    deleted_at = db.Column(db.DateTime, nullable=True)  # soft delete


# Zapytania użytkowników
class UserSearch(db.Model):
    __tablename__ = 'user_searches'
    id = db.Column(db.BigInteger, primary_key=True)
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'), nullable=False)
    text_searched = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())

# Foldery zapisane przez użytkowników
class SavedFolder(db.Model):
    __tablename__ = 'saved_folders'
    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(30), nullable=False)
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'), nullable=False)

# Publikacje
class Publication(db.Model):
    __tablename__ = 'publications'
    id = db.Column(db.BigInteger, primary_key=True)
    arxiv_id = db.Column(db.String(10), nullable=False)  # np. "0704.0001"
    title = db.Column(db.Text, nullable=False)
    language_id = db.Column(db.BigInteger, db.ForeignKey('languages.id'), nullable=False)
    abstract = db.Column(db.Text, nullable=False)
    origin_url = db.Column(db.Text, nullable=False)
    origin_url_html = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, server_default=func.now())
    updated_at = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

# Cytowania publikacji
class Citation(db.Model):
    __tablename__ = 'citations'
    id = db.Column(db.BigInteger, primary_key=True)
    origin_publication_id = db.Column(db.BigInteger, db.ForeignKey('publications.id'), nullable=False)
    citation_publication_id = db.Column(db.BigInteger, db.ForeignKey('publications.id'), nullable=False)



# Autorzy publikacji
class PublicationAuthor(db.Model):
    __tablename__ = 'publication_authors'
    id = db.Column(db.BigInteger, primary_key=True)
    first_name = db.Column(db.String(30), nullable=False)
    last_name = db.Column(db.String(30), nullable=False)
    country_id = db.Column(db.BigInteger, db.ForeignKey('countries.id'), nullable=False)

# Kategorie publikacji
class Category(db.Model):
    __tablename__ = 'categories'
    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(30), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())


class SeenPublication(db.Model):
    __tablename__ = 'seen_publications'
    id = db.Column(db.BigInteger, primary_key=True)
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'), nullable=False)
    publication_id = db.Column(db.BigInteger, db.ForeignKey('publications.id'), nullable=False)
    category_id = db.Column(db.BigInteger, db.ForeignKey('categories.id'), nullable=False)

# Relacja między kategoriami a publikacjami (tabela łącząca)
class CategoryPublication(db.Model):
    __tablename__ = 'categories_publications'
    id = db.Column(db.BigInteger, primary_key=True)
    category_id = db.Column(db.BigInteger, db.ForeignKey('categories.id'), nullable=False)
    publication_id = db.Column(db.BigInteger, db.ForeignKey('publications.id'), nullable=False)

# Zapisane publikacje w folderach
class SavedPublication(db.Model):
    __tablename__ = 'saved_publications'
    id = db.Column(db.BigInteger, primary_key=True)
    folder_id = db.Column(db.BigInteger, db.ForeignKey('saved_folders.id'), nullable=False)
    publication_id = db.Column(db.BigInteger, db.ForeignKey('publications.id'), nullable=False)

# Polubione publikacje
class LikedPublication(db.Model):
    __tablename__ = 'liked_publications'
    id = db.Column(db.BigInteger, primary_key=True)
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'), nullable=False)
    publication_id = db.Column(db.BigInteger, db.ForeignKey('publications.id'), nullable=False)

# Ocena przydatności publikacji
class HelpfulPublication(db.Model):
    __tablename__ = 'helpful_publications'
    id = db.Column(db.BigInteger, primary_key=True)
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'), nullable=False)
    publication_id = db.Column(db.BigInteger, db.ForeignKey('publications.id'), nullable=False)
    status = db.Column(db.Boolean, nullable=False)  # True - pomocna, False - nie

# Kraje
class Country(db.Model):
    __tablename__ = 'countries'
    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    iso_name = db.Column(db.String(3), nullable=False)

# Języki
class Language(db.Model):
    __tablename__ = 'languages'
    id = db.Column(db.BigInteger, primary_key=True)
    language_name = db.Column(db.String(60), unique=True, nullable=False)
    iso_name = db.Column(db.String(3), nullable=False)

# Relacja między krajami a językami (tabela łącząca)
class CountryLanguage(db.Model):
    __tablename__ = 'countries_languages'
    id = db.Column(db.BigInteger, primary_key=True)
    country_id = db.Column(db.BigInteger, db.ForeignKey('countries.id'), nullable=False)
    language_id = db.Column(db.BigInteger, db.ForeignKey('languages.id'), nullable=False)



# Utworzenie wszystkich tabel na podstawie zdefiniowanych modeli
with app.app_context():
    db.create_all()
    print("Database PostgreSQL created sucessfully")
if __name__ == '__main__':
    app.run(debug=True)

