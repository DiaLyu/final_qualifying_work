import {useState} from 'react';

import Spinner from '../Spinner/Spinner';

const QuestionForm = (props) => {
    const [userName, setUserName] = useState('');
    const [emailUser, setEmailUser] = useState('');
    const [descrError, setDescrError] = useState('');
    const [ready, setReady] = useState(true);
    const [successAnswer, setSuccessAnswer] = useState(false);

    const [loading, setLoading] = useState(false);           // загрузка данных с сервера
    const [error, setError] = useState(false);   

    const onUpdateName = (e) => {
        const nameUser = e.target.value;
        setUserName(nameUser)
    }

    const onUpdateEmail = (e) => {
        const emailUs = e.target.value;
        setEmailUser(emailUs);
    }

    const onUpdateDescr = (e) => {
        const descr = e.target.value;
        setDescrError(descr);
    }

    const closeModal = () =>{
        props.onQuestionVisib();
        setSuccessAnswer(false);
        setUserName('');
        setEmailUser('');
        setDescrError('');
        setError(false);
    }

    const emailSubmit = (e) => {
        e.preventDefault();
        console.log(e)

        const re = /^[\w-\.]+@[\w-]+\.[a-z]{2,4}$/i;
        let valid = re.test(emailUser);

        if (userName === '' || descrError === '' || !valid){
            setReady(false);
        } else{
            setReady(true);
            setLoading(loading => true);

            const data = new FormData();
            data.append('user_name', userName);
            data.append('email', emailUser);
            data.append('descr_error', descrError)

            fetch('http://localhost:5000/send_email', {
                method: 'POST',
                body: data,
            }).then((response) => {
                if (response.ok){
                    response.json().then((body) => {
                        console.log("body ", body);
                        setSuccessAnswer(body["success"]);
                    });
                    setLoading(loading => false);
                    setUserName('');
                    setEmailUser('');
                    setDescrError('');
                } else {
                    throw Error('Не удалось отправить форму')
                }
            }).catch((e) => {
                console.log("Error questionform ", e);
                setLoading(loading => false);
                setError(true);
                setUserName('');
                setEmailUser('');
                setDescrError('');
            });
        }
    }

    const View = (errorMessage) => {
        return (
            <>
                {
                    successAnswer ?
                        <div className="modal-win__title yellow-font">{errorMessage ? errorMessage : 'Вопрос отправлен. Благодарим за обратную связь!'}</div>
                    :
                        <>
                            <div className="modal-win__title yellow-font">Сообщить об ошибке</div>
                            <form action="#" className='form-error__form'  method="POST" onSubmit={emailSubmit}>
                                <div className="form-error__form__input">
                                    <input 
                                        type="text" 
                                        placeholder="Ваше имя"
                                        value={userName}
                                        style={userName || ready ? {} : {border: '2px solid red'}}
                                        onChange={onUpdateName}/>
                                </div>
                                <div className="form-error__form__input">
                                    <input 
                                        type="text" 
                                        placeholder="Введите email" 
                                        value={emailUser}
                                        style={emailUser || ready ? {} : {border: '2px solid red'}}
                                        onChange={onUpdateEmail}/>
                                </div>
                                <div className="form-error__form__input message-box">
                                    <textarea 
                                        placeholder="Описание ошибки"
                                        value={descrError}
                                        style={descrError || ready ? {} : {border: '2px solid red'}}
                                        onChange={onUpdateDescr}></textarea>
                                </div>
                                <div className="form-error__form__button">
                                    <input type="submit" value="Отправить" className='yellow-btn'/>
                                </div>
                            </form>
                        </>
                }
            </>
        )
    }


    const ViewError = () => {
        return (
            <div className="modal-win__title yellow-font">Не удалось отправить форму. Напишите позже</div>
        )
    }
    
    
    const errorMessage = error ? ViewError() : null;
    const spinner = loading ? <Spinner/> : null;
    const content = !(loading || error) ? View(errorMessage) : null;

    return (
        <div className={`form-error modal-win ${props.questionVisib ? '' : 'hide'}`}>
            <div className="form-error__container modal-win__cont">
                <div onClick={closeModal} className="close"></div>
                {errorMessage}
                {spinner}
                {content}
            </div>
        </div>
    )
}

export default QuestionForm;