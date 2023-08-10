import {useState} from 'react';

import ControlPanel from '../controlPanel/ControlPanel';
import BookDownload from '../bookDownload/BookDownload';

import book_icon from '../../resources/img/openbook.png';
import download_btn from '../../resources/img/download1.png';

const MapMenu = () => {

    const [activeBooks, setActiveBooks] = useState('hide');
    const [activeDown, setActiveDown] = useState('hide');
    const [filePath, setFilePath] = useState(null);

    const onUpdateFile = (filePath) => {
        setFilePath(filePath);
    }

    const openPanel = (state1, setState1, state2, setState2) => {
        if(state1 === 'hide'){
            setState1('');
            if(state2 === ''){
                setState2('hide');
            }
        } else {
            setState1('hide');
        }
    }

    return (
        <>
            <div className="main__menu">
                <ul className="main__menu__items">
                    <li onClick={() => openPanel(activeBooks, setActiveBooks, activeDown, setActiveDown)} className="menu__item" style={activeBooks === '' ? {'background' : '#ac8a6d'} : {'background' : 'none'}}>
                        <img src={book_icon} alt="Иконка книги" className="menu__item__book" />
                    </li>
                    <li onClick={() => openPanel(activeDown, setActiveDown, activeBooks, setActiveBooks)} className="menu__item" style={activeDown === '' ? {'background' : '#ac8a6d'} : {'background' : 'none'}}>
                        <img src={download_btn} alt="Иконка загрузки" className="menu__item__download" />
                    </li>
                </ul>
            </div>

            <div className="main__map">
                <BookDownload activeDown={activeDown} onUpdateFile={onUpdateFile}/>
                <ControlPanel activeBooks={activeBooks} filePath={filePath}/>
            </div>
        </>
    )
}

export default MapMenu;