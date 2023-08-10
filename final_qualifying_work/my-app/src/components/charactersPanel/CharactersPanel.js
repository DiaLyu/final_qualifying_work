import {useState, useEffect} from 'react';

import RouteService from '../../service/RouteService';
import ErrorMessage from '../errorMessage/ErrorMessage';
import Spinner from '../Spinner/Spinner'

const CharactersPanel = (props) => {
    const {dataItem} = props;
    const [loading, setLoading] = useState(true);           // загрузка данных с сервера
    const [error, setError] = useState(false);              // возникновение ошибки с сервера
    const [selectedInput, setSelectedInput] = useState([]);

    const [characterList, setCharacterList] = useState([])  // список персонажей из выбранной книги

    const routeService = new RouteService();

    useEffect(() => {
        setLoading(loading => true);
        fetchCharacters();
    }, [props.idBook])

    const fetchCharacters = () => {
        const {idBook} = props;
        if (!idBook) {
            return;
        }
        
        routeService.getBookCharacters(idBook)
            .then(onCharacterListLoaded)
            .catch(onError)
    }
    
    const onCharacterListLoaded = (newBookList) =>{
        setCharacterList(newBookList);
        setLoading(loading => false);
        setSelectedInput([])
    }

    const onError = () => {
        setError(true);
        setLoading(loading => false);
    }

    const handleChange = (e, id) => {
        let proba = Array.from(selectedInput);
        console.log(proba)
        if (proba.length === 0){
            proba.push([id, e.target.checked]);
        } else {
            let flag = true;
            for(let i = 0; i < proba.length; i++){
                if (proba[i][0] === id){
                    proba[i][1] = e.target.checked;
                    flag = false;
                }
            }
            if (flag) {
                proba.push([id, e.target.checked]);
            }
        }
        setSelectedInput(proba);
        props.onUpdateSelected(proba);

        console.log("CharactersPanel selectedInput ", selectedInput)
    };

    const elements = characterList.map((item, i) => {
        return (
            <li key={item.id_book_charact.toString()}>
                <div></div>
                <input type="checkbox" id={`hero_${item.id_book_charact}`} className='checkbox-hero' onChange={(e) => handleChange(e, item.id_book_charact)}/>
                <label htmlFor={`hero_${item.id_book_charact}`}>{item.name_character}</label>
            </li>
        )
    });


    const errorMessage = error ? <ErrorMessage/> : null;
    const spinner = loading ? <Spinner/> : null;
    const content = !(loading || error) ? elements : null;

    return (
        <div className={`control-panel__characters ${dataItem === "" ? 'hide': ''}`}>
            <div className="name-book">{dataItem}</div>
            <form className='control-panel__characters__item'>
                <fieldset>
                    <legend>Маршруты героев</legend>
                    <ul>
                        {errorMessage}
                        {spinner}
                        {content}
                    </ul>
                </fieldset>
            </form>
        </div>
    )
}

export default CharactersPanel;