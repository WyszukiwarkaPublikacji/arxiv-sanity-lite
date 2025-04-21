from sqlalchemy.orm import declarative_base
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, Boolean,
    DateTime, LargeBinary, ForeignKey, func, Integer
)
import os, sys
# reuse the existing metadata and engine


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)


from SQLLiteAlchemyInstance import SQLAlchemyInstance

metadata = SQLAlchemyInstance().get_sqllite_metadata()
engine = SQLAlchemyInstance().get_engine()
# Declarative base bound to the existing metadata
Base = declarative_base(metadata=metadata)

class Users(Base):
    __tablename__ = 'users'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    name = Column(String(30), nullable=False, unique=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class UserSearches(Base):
    __tablename__ = 'user_searches'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=False)
    text_searched = Column(String(120), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class SavedFolders(Base):
    __tablename__ = 'saved_folders'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    name = Column(String(30), nullable=False)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=False)

class Publications(Base):
    __tablename__ = 'publications'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    arxiv_id = Column(String(10), nullable=False)
    title = Column(Text, nullable=False)
    language_id = Column(BigInteger, ForeignKey('languages.id'), nullable=False)
    abstract = Column(Text, nullable=False)
    origin_url = Column(Text, nullable=False)
    origin_url_html = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class PublicationAuthors(Base):
    __tablename__ = 'publication_authors'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    first_name = Column(String(30), nullable=False)
    last_name = Column(String(30), nullable=False)
    country_id = Column(BigInteger, ForeignKey('countries.id'), nullable=False)

class Categories(Base):
    __tablename__ = 'categories'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    name = Column(String(30), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class SeenPublications(Base):
    __tablename__ = 'seen_publications'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    origin_publication_id = Column(String(30), ForeignKey('papers.key'), nullable=False)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=False)

class CategoryPublications(Base):
    __tablename__ = 'categories_publications'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    category_id = Column(BigInteger, ForeignKey('categories.id'), nullable=False)
    publication_id = Column(BigInteger, ForeignKey('publications.id'), nullable=False)

class SavedPublications(Base):
    __tablename__ = 'saved_publications'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    folder_id = Column(BigInteger, ForeignKey('saved_folders.id'), nullable=False)
    publication_id = Column(BigInteger, ForeignKey('publications.id'), nullable=False)

class LikedPublications(Base):
    __tablename__ = 'liked_publications'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=False)
    publication_id = Column(BigInteger, ForeignKey('publications.id'), nullable=False)

class HelpfulPublications(Base):
    __tablename__ = 'helpful_publications'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=False)
    publication_id = Column(BigInteger, ForeignKey('publications.id'), nullable=False)
    status = Column(Boolean, nullable=False)

class Countries(Base):
    __tablename__ = 'countries'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    iso_name = Column(String(3), nullable=False)

class Languagies(Base):
    __tablename__ = 'languages'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    language_name = Column(String(60), unique=True, nullable=False)
    iso_name = Column(String(3), nullable=False)

class CountryLanguagies(Base):
    __tablename__ = 'countries_languages'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True)
    country_id = Column(BigInteger, ForeignKey('countries.id'), nullable=False)
    language_id = Column(BigInteger, ForeignKey('languages.id'), nullable=False)

class MetaDataDB(Base):
    __tablename__ = 'metas'
    __table_args__ = {'sqlite_autoincrement': True}
    key = Column(String(30), primary_key=True)
    value = Column(LargeBinary, nullable=True)

class Papers(Base):
    __tablename__ = 'papers'
    __table_args__ = {'sqlite_autoincrement': True}
    key = Column(String(30), primary_key=True)
    value = Column(LargeBinary, nullable=True)

class Citation(Base):
    __tablename__ = 'citations'
    __table_args__ = {'sqlite_autoincrement': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    origin_publication_id = Column(String(30), ForeignKey('papers.key'), nullable=False)
    citation_publication_id = Column(String(30), ForeignKey('papers.key'), nullable=False)

# To create all tables:

def  creation():
    Base.metadata.create_all(engine)
    print("All tables created")

def  creation_with_drop():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(engine)
    print("All tables created")


# creation_with_drop()