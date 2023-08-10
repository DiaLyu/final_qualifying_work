import {useState, useEffect} from 'react';

import SearchPanel from '../searchPanel/SearchPanel';
import CharactersPanel from '../charactersPanel/CharactersPanel';
import RouteService from '../../service/RouteService';
import ErrorMessage from '../errorMessage/ErrorMessage';
import Spinner from '../Spinner/Spinner';
import MapRoutes from '../mapRoutes/MapRoutes';


const ControlPanel = (props) => {
    const {activeBooks, filePath} = props;
    const [loading, setLoading] = useState(true);           // загрузка данных с сервера
    const [error, setError] = useState(false);              // возникновение ошибки с сервера

    const [dataItem, setDataItem] = useState("");           // название выбранной книги
    const [idBook, setIdBook] = useState(null);             // id выбранной книги
    const [term, setTerm]  = useState('');                  // поисковая строка
    const [bookList, setBookList] = useState([]);           // список полученных из базы данных книг
    const [selectedInput, setSelectedInput] = useState([]); // выбранные персонажи

    const routeService = new RouteService();

    useEffect(() => {
        fetchBooks();
    }, [filePath])

    const fetchBooks = () => {
        routeService.getAllBooks()
            .then(onBookListLoaded)
            .catch(onError)
    }

    const onBookListLoaded = (newBookList) =>{
        setBookList(newBookList);
        setLoading(loading => false);
    }

    const onError = () => {
        setError(true);
        setLoading(loading => false);
    }

    const onChangeIdBook = (name, id) => {
        setDataItem(name);
        setIdBook(id);
        setSelectedInput([]);
        console.log("onChangeIdBook")
    }

    const onUpdateSearch = (term) => {
        setTerm(term);
    }

    const onUpdateSelected = (selectedChar) => {
        setSelectedInput(selectedChar);
    }

    const searchEmp = (items, term) => {
        if (term.length === 0){
            return items;
        } 

        return items.filter(item => {
            return item.name_book.toLowerCase().indexOf(term.toLowerCase()) > -1
        })
    }

    const visibleData = searchEmp(bookList, term);
    const elements = visibleData.map(item => {
            return (
                <>
                    <li key={item.id_book.toString()}
                        onClick={() => onChangeIdBook(item.name_book, item.id_book)}
                        className="control-panel__book" 
                        >
                        <div>
                            {item.name_book}
                        </div> 
                    </li>
                    <hr className='control-panel__line'/>
                </>
            )
        }
    )

    const errorMessage = error ? <ErrorMessage/> : null;
    const spinner = loading ? <Spinner/> : null;
    const content = !(loading || error) ? elements : null;


    return (
        <>
            <div className={`control-panel ${activeBooks}`}>
                <div className="control-panel__searchbooks">
                    <div className="control-panel__search">
                        <SearchPanel onUpdateSearch={onUpdateSearch}/>
                    </div>
                    <div className="control-panel__books">
                        <ul>
                            {errorMessage}
                            {spinner}
                            {content}
                        </ul>
                    </div>
                </div>

                <CharactersPanel dataItem={dataItem} idBook={idBook} onUpdateSelected={onUpdateSelected}/>
            </div>

            <MapRoutes selectedInput={selectedInput}/>
        </>
    )
}



export default ControlPanel;