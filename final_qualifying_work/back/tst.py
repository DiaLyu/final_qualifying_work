from flask import Flask, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import psycopg2

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:12345@localhost/character_routes?client_encoding=utf8'
db = SQLAlchemy(app)
CORS(app)


class Book(db.Model):
    id_book = db.Column(db.Integer, primary_key=True, nullable=False)
    name_book = db.Column(db.String(100))
    path_file = db.Column(db.String(100))

    books = db.relationship('Book_Character', backref='book_conn', lazy='dynamic')

    def __repr__(self):
        return f"Book: {self.name_book}"
    
    def __init__(self, name_book, path_file):
        self.name_book = name_book
        self.path_file = path_file


class Character(db.Model):
    id_character = db.Column(db.Integer, primary_key=True, nullable=False)
    name_character = db.Column(db.String(100))
    descr = db.Column(db.Text)

    characters = db.relationship('Book_Character', backref='charact_conn', lazy='dynamic')

    def __repr__(self):
        return f"Character: {self.name_character} {self.descr}"
    
    def __init__(self, name_character, descr):
        self.name_character = name_character
        self.descr = descr


class Book_Character:
    id_book_charact = db.Column(db.Integer, primary_key=True, nullable=False)
    id_character = db.Column(db.Integer, db.ForeignKey('character.id_character'))
    id_book = db.Column(db.Integer, db.ForeignKey('book.id_book'))

    route = db.relationship('Route_point', backref='book_charact')

    def __init__(self, id_character, id_book):
        self.id_character = id_character
        self.id_book = id_book


class Country:
    id_country = db.Column(db.Integer, primary_key=True, nullable=False)
    name_country = db.Column(db.String(50))

    last_names_country = db.relationship('Last_Name_Country', backref='act_country')
    places = db.relationship('Place', backref='country')

    def __repr__(self):
        return f"Country: {self.name_country}"
    
    def __init__(self, name_country):
        self.name_country = name_country


class Last_Name_Country:
    id_last_country = db.Column(db.Integer, primary_key=True, nullable=False)
    id_country = db.Column(db.Integer, db.ForeignKey('country.id_country'), nullable=False)
    last_country = db.Column(db.String(50))

    def __repr__(self):
        return f"Last Name Country: {self.last_country} \n Country: {self.id_country}"
    
    def __init__(self, id_country, last_country):
        self.id_country = id_country
        self.last_country = last_country


class Place:
    id_place = db.Column(db.Integer, primary_key=True, nullable=False)
    id_country = db.Column(db.Integer, db.ForeignKey('country.id_country'), nullable=False)
    name_place = db.Column(db.String(100))

    last_names_place = db.relationship('Last_Name_Place', backref='act_place')
    place = db.relationship('Coords', backref='place')
    route_points = db.relationship('Route_point', backref='place')

    def __repr__(self):
        return f"Name Place: {self.name_place} \n Country: {self.id_country}"
    
    def __init__(self, id_country, name_place):
        self.id_country = id_country
        self.name_place = name_place


class Last_Name_Place:
    id_last_place = db.Column(db.Integer, primary_key=True, nullable=False)
    id_place = db.Column(db.Integer, db.ForeignKey('place.id_place'), nullable=False)
    last_place = db.Column(db.String(100))
    
    def __repr__(self):
        return f"Last Name Place: {self.last_place} \n Country: {self.id_country}"
    
    def __init__(self, id_place, last_place):
        self.id_place = id_place
        self.last_place = last_place


class Coords:
    id_coords = db.Column(db.Integer, primary_key=True, nullable=False)
    id_place = db.Column(db.Integer, db.ForeignKey('place.id_place'), nullable=False)
    longitude = db.Column(db.Integer)
    latitude = db.Column(db.Integer)
    
    def __repr__(self):
        return f"longitude: {self.longitude} \n latitude: {self.latitude}"
    
    def __init__(self, id_place, longitude, latitude):
        self.id_place = id_place
        self.longitude = longitude
        self.latitude = latitude


class Route_point:
    id_point = db.Column(db.Integer, primary_key=True, nullable=False)
    id_book_charact = db.Column(db.Integer, db.ForeignKey('book_character.id_book_charact'), nullable=False)
    id_place = db.Column(db.Integer, db.ForeignKey('place.id_place'), nullable=False)
    order_in = db.Column(db.Integer)

    def __init__(self, id_book_charact, id_place, order_in):
        self.id_book_charact = id_book_charact
        self.id_place = id_place
        self.order_in = order_in