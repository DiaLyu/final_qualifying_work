import {useState} from 'react';

import search_btn from '../../resources/img/find_icon.png';

const SearchPanel = (props) => {
    const [term, setTerm] = useState('');

    const onUpdateSearch = (e) => {
        const term = e.target.value;
        setTerm(term);
        props.onUpdateSearch(term);
    }

    return (
        <form className='control-panel__search__container'>
            <input 
                type="text" 
                placeholder="Война и мир" 
                value={term}
                onChange={onUpdateSearch}/>
            <img src={search_btn} alt="Поиск" className="control-panel__search__button" />
        </form>
    )
}

export default SearchPanel;