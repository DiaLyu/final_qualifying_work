

class RouteService {
    _apiBase = 'http://localhost:5000/';

    getResource = async (url) => {
        let res = await fetch(url);
     
        if (!res.ok) {
            throw new Error(`Could not fetch ${url}, status: ${res.status}`);
        }
    
        return await res.json();
    }

    getAllBooks = async () => {
        const res = await this.getResource(`${this._apiBase}books`);
        return res.books;
    }

    getBookCharacters = async (id) => {
        const res = await this.getResource(`${this._apiBase}book_characters/${id}`);
        return res.book_characters;
    }

    getRouteCharacter = async (id) => {
        const res = await this.getResource(`${this._apiBase}route_book/${id}`);
        return {
            "id": id,
            "route": res.route
        }
    }
}

export default RouteService;