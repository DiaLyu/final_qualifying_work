import {useState, useEffect} from 'react';
import {Link} from 'react-router-dom';

import logo from '../../resources/img/logo.png';
import burger from '../../resources/img/burger.png';
import arrow from '../../resources/icons/arrow-left.png';
import question from '../../resources/icons/book-question.png';
import cloud from '../../resources/icons/cloud.png';
import share from '../../resources/icons/share.png';
import print from '../../resources/icons/print.png';

const Header = (props) => {
    const [openMenu, setOpenMenu] = useState(false);

    const onChangeVisibility = () => {
        setOpenMenu(!openMenu);
    }

    const onQuestionVisib = () => {
        setOpenMenu(false);
        props.onQuestionVisib();
    }

    const onHelpVisib = () => {
        setOpenMenu(false);
        props.onHelpVisib();
    }

    return(
        <header className="header">
                <div className="header__logo">
                    <Link to='/'> 
                        <img className="header__logo__img" src={logo} alt="Логотип" />
                    </Link>
                </div>
                <div className="header__menu">
                    <div onClick={onChangeVisibility} className="header__menu__icon">
                        <img src={burger} alt="burger menu" />
                    </div>
                    <div className={`header__menu__list ${openMenu ? '' : 'hide'}`}>
                        <ul>
                            <li onClick={onQuestionVisib} className="header__menu__item">
                                <div className='header__menu__img'>
                                    <img src={question} alt="Иконка для вопроса" />
                                </div>
                                <div className="header__menu__subheader">Сообщать об ошибке</div>
                            </li>
                            <li onClick={onHelpVisib} className="header__menu__item">
                                <div className='header__menu__img'>
                                    <img src={cloud} alt="Иконка справки" />
                                </div>
                                <div className="header__menu__subheader">Справка</div>
                            </li>
                            <li className="header__menu__item">
                                <Link to='/'>
                                    <div className='header__menu__img'>
                                        <img src={arrow} alt="Иконка возврата" />
                                    </div>
                                    <div className="header__menu__subheader">На главную</div>
                                </Link>
                            </li>
                        </ul>
                    </div>
                </div>
            </header>
    )
}

export default Header;