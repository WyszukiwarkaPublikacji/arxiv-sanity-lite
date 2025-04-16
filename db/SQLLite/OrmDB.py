import sys
import os
import sqlalchemy as db_alchemy
from sqlalchemy import Table, Column, Integer, BigInteger, String, Text, Boolean, DateTime, LargeBinary, ForeignKey, MetaData, func


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)


from SQLLiteAlchemyInstance import SQLAlchemyInstance

metadata = SQLAlchemyInstance().get_sqllite_metadata()
engine = SQLAlchemyInstance().get_engine()

Users = Table('users', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('first_name', String(30), nullable=False, unique=True),
    Column('last_name', String(30), nullable=False, unique=True),
    Column('login', String(30), nullable=False, unique=True),
    Column('email', String(120), nullable=False, unique=True),
    Column('password', String(255), nullable=False),
    Column('date_birth', DateTime, nullable=False),
    Column('created_at', DateTime, server_default=func.now()),
    Column('updated_at', DateTime, server_default=func.now(), onupdate=func.now()),
    Column('deleted_at', DateTime, nullable=True)
)

UserSearches = Table('user_searches', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('user_id', BigInteger, ForeignKey('users.id'), nullable=False),
    Column('text_searched', String(120), nullable=False),
    Column('created_at', DateTime, server_default=func.now())
)

SavedFolders = Table('saved_folders', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('name', String(30), nullable=False),
    Column('user_id', BigInteger, ForeignKey('users.id'), nullable=False)
)

Publications = Table('publications', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('arxiv_id', String(10), nullable=False),
    Column('title', Text, nullable=False),
    Column('language_id', BigInteger, ForeignKey('languages.id'), nullable=False),
    Column('abstract', Text, nullable=False),
    Column('origin_url', Text, nullable=False),
    Column('origin_url_html', Text),
    Column('created_at', DateTime, server_default=func.now()),
    Column('updated_at', DateTime, server_default=func.now(), onupdate=func.now())
)

Citations = Table('citations', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('origin_publication_id', BigInteger, ForeignKey('publications.id'), nullable=False),
    Column('citation_publication_id', BigInteger, ForeignKey('publications.id'), nullable=False)
)

PublicationAuthors = Table('publication_authors', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('first_name', String(30), nullable=False),
    Column('last_name', String(30), nullable=False),
    Column('country_id', BigInteger, ForeignKey('countries.id'), nullable=False)
)

Categories = Table('categories', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('name', String(30), unique=True, nullable=False),
    Column('created_at', DateTime, server_default=func.now())
)

SeenPublications = Table('seen_publications', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('user_id', BigInteger, ForeignKey('users.id'), nullable=False),
    Column('publication_id', BigInteger, ForeignKey('publications.id'), nullable=False),
    Column('category_id', BigInteger, ForeignKey('categories.id'), nullable=False)
)

CategoryPublications = Table('categories_publications', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('category_id', BigInteger, ForeignKey('categories.id'), nullable=False),
    Column('publication_id', BigInteger, ForeignKey('publications.id'), nullable=False)
)

SavedPublications = Table('saved_publications', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('folder_id', BigInteger, ForeignKey('saved_folders.id'), nullable=False),
    Column('publication_id', BigInteger, ForeignKey('publications.id'), nullable=False)
)

LikedPublications = Table('liked_publications', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('user_id', BigInteger, ForeignKey('users.id'), nullable=False),
    Column('publication_id', BigInteger, ForeignKey('publications.id'), nullable=False)
)

HelpfulPublications = Table('helpful_publications', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('user_id', BigInteger, ForeignKey('users.id'), nullable=False),
    Column('publication_id', BigInteger, ForeignKey('publications.id'), nullable=False),
    Column('status', Boolean, nullable=False)
)

Countries = Table('countries', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('name', String(80), nullable=False),
    Column('iso_name', String(3), nullable=False)
)

Languages = Table('languages', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('language_name', String(60), unique=True, nullable=False),
    Column('iso_name', String(3), nullable=False)
)

CountryLanguages = Table('countries_languages', metadata,
    Column('id', BigInteger, primary_key=True),
    Column('country_id', BigInteger, ForeignKey('countries.id'), nullable=False),
    Column('language_id', BigInteger, ForeignKey('languages.id'), nullable=False)
)

MetaDataDB = Table('metas', metadata,
    Column('key', String(30), primary_key=True),
    Column('value', LargeBinary, nullable=True)
)

Papers = Table('papers', metadata,
    Column('key', String(30), primary_key=True),
    Column('value', LargeBinary, nullable=True)
)


def main():
  metadata.drop_all(engine)
  metadata.create_all(engine)

if __name__ == '__main__':
  main()

