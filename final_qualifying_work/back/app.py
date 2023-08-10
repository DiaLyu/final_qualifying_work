#-*- coding: utf-8 -*-
import os
from flask import Flask, request, render_template, flash, redirect, url_for, session
from flask_cors import CORS
import psycopg2
from werkzeug.utils import secure_filename
import logging
from analyzeBook.analyze import AnalyzeText
from geopy.geocoders import Nominatim
from flask_mail import Message, Mail

UPLOAD_FOLDER = './file_book'
ALLOWED_EXTENSIONS = set(['txt', 'pdf'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('HELLO WORLD')

app = Flask(__name__)
mail = Mail(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAIL_SERVER']='smtp.yandex.ru'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'CalininaSvet12@yandex.ru'
app.config['MAIL_PASSWORD'] = '****'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)
CORS(app)

def get_db_connection():
    conn = psycopg2.connect(host='localhost',
                            database='character_routes',
                            user='postgres',
                            password='12345')
    return conn

def format_book(book):
    return {
        "id_book": book[0],
        "name_book": book[1],
        "path_file": book[2]
    }

def format_book_charact(books):
    return {
        "id_book_charact": books.id_book_charact,
        "name_book": books.name_book,
        "name_character": books.name_character
    }

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def add_route_entries(filename, filepath, routes, coords):
    conn = get_db_connection()
    cur = conn.cursor()

    names = [names['Name'] for names in routes]
    route = [route['Route'] for route in routes]
    descr = [descr['Descr'] for descr in routes]

    print("filename ", filename)
    print("------------")
    print("filepath ", filepath)
    print("------------")
    print("names ", names)
    print("------------")
    print("route ", route)
    print("------------")
    print("coords ", coords)
    print("------------")

    message = ""

    if names != [] and coords != []:
    # добавление книги 
        cur.execute(f"INSERT INTO book (name_book, path_file) VALUES ('{filename}', '{filepath}') RETURNING id_book;")
        id_book = cur.fetchall()
        conn.commit()

        # добавление персонажей
        values = ""
        for i in range(len(names)):
            values += "('" + names[i] + "', '" + descr[i] + "'), "
        cur.execute(f"INSERT INTO character (name_character, descr) VALUES {values[:-2]} RETURNING id_character;")
        id_characters = list(cur.fetchall())
        print("id_characters ", id_characters)
        conn.commit()

        # добавление связи между книгой и персонажами
        values = ""
        for id in id_characters:
            values += "(" + str(id_book[0][0]) + ", " + str(id[0]) + "), "
        cur.execute(f"INSERT INTO book_character (id_book, id_character) VALUES {values[:-2]} RETURNING id_book_charact;")
        id_book_charact = list(cur.fetchall())
        print("id_book_charact ", id_book_charact)
        conn.commit()

        # добавление координат городов, если их нет
        cities = [city['city'] for city in coords]
        values = ""
        for city in cities:
            values += "LOWER(place.name_place) = '" + city + "' OR " + "LOWER(last_name_place.last_place) = '" + city + "' OR "
        cur.execute(f"""SELECT place.id_place, LOWER(place.name_place), LOWER(last_name_place.last_place)
                        FROM place 
                        LEFT JOIN last_name_place ON last_name_place.id_place = place.id_place
                        WHERE {values[:-4]};""")
        place = cur.fetchall() # получение id мест
        print("place ", place)

        id_places = [data[0] for data in place]
        if id_places != []:
            values = ""
            for id in id_places:
                values += "id_place = " + str(id) + " OR "
            cur.execute(f'SELECT id_place FROM coords WHERE {values[:-4]}')
            place_coords = cur.fetchall() # получение id мест
            print("place_coords ", place_coords)

        id_coords = [data[0] for data in place_coords]
        print("id_coords ", id_coords)

        not_in_coords = list(set(id_places) ^ set(id_coords))

        data_coords = []
        if not_in_coords != []:
            not_in_coords = [pl for coor in not_in_coords for pl in place if pl[0] == coor]
            print("not_in_coords ", not_in_coords)
            data_coords = [{'id_place': city[0], 'longitude': data['coords'][1], 'latitude': data['coords'][0]} for data in coords for city in not_in_coords if data['city'] == city[1] or data['city'] == city[2]]
            print("data_coords ", data_coords)

            values = ""
            for data in data_coords:
                values += '(' + str(data['id_place']) + ', ' + str(data['longitude']) + ', ' + str(data['latitude']) + '),'
            cur.execute(f"INSERT INTO coords (id_place, longitude, latitude) VALUES {values[:-1]}")
            conn.commit()

        route_point = []

        for i in range(len(id_book_charact)):
            id_cities = []
            for rt in route[i]:
                print("rt ", rt)
                id_city = [pl[0] for pl in place if pl[1] == rt or pl[2] == rt]
                print(i, " id_city ", id_city)
                id_cities.append(id_city[0])
            route_point.append({'id_book_charact': id_book_charact[i], 'id_places': id_cities})

        values = ""
        for data in route_point:
            for i in range(len(data['id_places'])):
                values += "(" + str(data['id_book_charact'][0]) + ', ' + str(data['id_places'][i]) + ', ' + str(i) + '),'

        cur.execute(f"INSERT INTO route_point (id_book_charact, id_place, order_in) VALUES {values[:-1]}")
        conn.commit()

        message = "Данные добавлены в базу данных"
    
    else:
        message = "Данные не обнаружены"

    cur.close()
    conn.close()
    return message

def get_fst_city(city):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(f"""SELECT place.id_place, LOWER(place.name_place), LOWER(last_name_place.last_place)
                        FROM place 
                        LEFT JOIN country ON country.id_country = place.id_country 
                        LEFT JOIN last_name_place ON place.id_place = last_name_place.id_place
                        WHERE LOWER(country.name_country) = '{city}'""")
    country_places = cur.fetchall()
    place = ""
    if len(country_places) != 0:
        place = country_places[0][1]

    cur.close()
    conn.close()

    return place


@app.route('/upload', methods=['POST'])
def fileUpload():
    if request.method == 'POST':
        userNameBook = request.values.get('filename')
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print("-----start -------")
            print("userNameBook ", userNameBook)
            routes = AnalyzeText(file_path).analyze()
            print(routes)
            print("----- end analyze -------")

            location1= []
            for i in range(len(routes)):
                route = []
                for city in routes[i]['Route']:
                    place = get_fst_city(city)
                    route.append(place if (place != "") else city)
                routes[i]['Route'] = route

            for nm_ct in routes:
                for city in nm_ct['Route']:
                    try:
                        geolocator = Nominatim(user_agent="http://localhost:3000/map")
                        location = geolocator.geocode(city)
                        location1.append({"city": city,"coords": list([location.latitude, location.longitude])})
                    except:
                        print("city", city)

            coords = list({v["city"]:v for v in location1}.values())
            print("coords ", coords)

            message = add_route_entries(userNameBook, file_path, routes, coords)

        return {'success': True, 'message': message}
    

@app.route('/send_email', methods=['POST'])
def sendEmail():
    if request.method == 'POST':
        userName = request.values.get('user_name')
        email = request.values.get('email')
        descr_error = request.values.get('descr_error')
        print("userNameBook ", userName)
        print("email ", email)
        print("descr_error ", descr_error)

        msg = Message(
                'Question from Book Trip',
                sender ='CalininaSvet12@yandex.ru',
                recipients = ['CalininaSvet12@yandex.ru']
               )
        msg.body = 'Имя пользователя: {}\nEmail: {}\nQuestion: {}'.format(userName, email, descr_error)
        mail.send(msg)

        return {'success': True}

@app.route('/', methods=['GET'])
def hello():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM book;')
    books = cur.fetchall()
    cur.close()
    conn.close()
    book_list = []
    for book in books:
        book_list.append(format_book(book))
    return book_list

# get all countries
@app.route('/books', methods=['GET'])
def get_books():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM book;')
    books = cur.fetchall()
    cur.close()
    conn.close()
    book_list = []
    for book in books:
        book_list.append(format_book(book))
    return {"books": book_list}


# get single book
@app.route('/books/<id>', methods = ['GET'])
def get_book(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM book WHERE id_book = {id};')
    books = cur.fetchall()
    cur.close()
    conn.close()
    return {'book': books[0]}


# get all characters from book with selected id
@app.route('/book_characters/<id>', methods=['GET'])
def get_charact_book(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""SELECT book_character.id_book_charact, character.name_character, book.name_book
    FROM book_character 
    JOIN book ON book.id_book = book_character.id_book 
    JOIN character ON character.id_character = book_character.id_character
    WHERE book.id_book = {id};""")
    book_characters = cur.fetchall()
    cur.close()
    conn.close()
    book_list = []
    for book in book_characters:
        book_list.append({
            "id_book_charact": book[0],
            "name_character": book[1],
            "name_book": book[2]    
        })
    return {"book_characters": book_list}


# get all routes
@app.route('/route_book/<id>', methods=['GET'])
def get_routes(id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""SELECT route_point.id_point, place.name_place, coords.longitude, coords.latitude, route_point.order_in FROM route_point
                    JOIN book_character ON book_character.id_book_charact = route_point.id_book_charact
                    JOIN place ON place.id_place = route_point.id_place
                    JOIN coords ON coords.id_place = place.id_place
                    WHERE route_point.id_book_charact = {id}""")
    route = cur.fetchall()
    cur.close()
    conn.close()
    points_list = []
    for book in route:
        points_list.append({
            "id_point": book[0],
            "name_place": book[1],
            "longitude": book[2],
            "latitude": book[3],
            "order_in": book[4]
        })
    points_list.sort(key=lambda x: x["order_in"])
    return {"route": points_list}


if __name__ == '__main__':
    app.run()
