import ErrorMessage from "../errorMessage/ErrorMessage";
import {Link} from 'react-router-dom';
import { Helmet } from "react-helmet";

import logo from '../../resources/img/logo.png';

const Page404 = () => {
    return (
        <div>
            <Helmet>
                <meta
                    name="description"
                    content="Эта страница не найдена"
                />
                <title>Страница не найдена</title>
            </Helmet>

            <header className="header">
                <div className="header__logo">
                    <Link to='/'>
                        <img className="header__logo__img" src={logo} alt="Логотип" />
                    </Link>
                </div>
            </header>
            <ErrorMessage/>
            <p style={{'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '24px'}}>Страница не найдена</p>
            <Link style={{'display': 'block', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '24px', 'marginTop': '30px'}} to="/">Вернуться на главную</Link>
        </div>
    )
}

export default Page404;