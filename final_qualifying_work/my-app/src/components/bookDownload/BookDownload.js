import {useState, useRef} from 'react';

import Spinner from '../Spinner/Spinner';

const BookDownload = (props) => {
    const [drag, setDrag] = useState(false);
    const [nameFile, setNameFile] = useState("Максимум 10мб");
    const [file, setFile] = useState(null)
    const [nameBook, setNameBook] = useState('');
    const [ready, setReady] = useState(true);
    const [error, setError] = useState(false);
    const [loading, setLoading] = useState(false);

    const inputEl = useRef(null);
    const {activeDown} = props;

    const dragStartHandler = (e) => {
        e.preventDefault();
        setDrag(true);
    }

    const dragLeaveHandler = (e) => {
        e.preventDefault();
        setDrag(false);
    }

    const onDropHandler = (e) => {
        e.preventDefault();
        let files = [...e.dataTransfer.files];
        setNameFile(files[0].name)
        setFile(files[0])
        setError(false);
        setDrag(false);
    }

    const onUpdateName = (e) => {
        const book = e.target.value;
        setNameBook(book);
    }

    const onChangeFile = () => {
        setNameFile(inputEl.current.files[0].name)
        setFile(inputEl.current.files[0])
        setError(false);
    }

    const handleSubmit = (e) => {
        e.preventDefault();
        console.log(e)

        if (nameBook === '' || file === null){
            setReady(false);
        } else {
            setLoading(loading => true);
            setReady(true);

            const data = new FormData();
            data.append('file', file);
            data.append('filename', nameBook);

            fetch('http://localhost:5000/upload', {
                method: 'POST',
                body: data,
            }).then((response) => {
                if (response.ok){
                    response.json().then((body) => {
                        console.log("body ", body)
                    });
                    setNameBook('');
                    setNameFile("Анализ окончен");
                    setFile(null);
                    setLoading(loading => false);
                } else {
                    throw Error('Не удалось провести анализ. Попробуйте позже')
                }
                props.onUpdateFile(file);
            }).catch((e) => {
                setNameBook('');
                setNameFile(String(e));
                setError(true);
                setFile(null);
                setLoading(loading => false);
            });
        }
    }

    return (
        <div className={`download-panel ${activeDown}`}>

            <div className="name-book">Новая книга</div>

            {
                loading ? 
                <>
                    <Spinner/>
                    <span className="input-file__text">Анализ текста...</span>
                </>
                :
                <form className="my-form" method="POST" onSubmit={handleSubmit}>
                    <label htmlFor="nameBook">Автор и название книги</label>
                    <input
                        id="nameBook"
                        type="text" 
                        placeholder='Л.Н. Толстой "Война и мир"' 
                        value={nameBook}
                        style={nameBook || ready ? {} : {border: '2px solid red'}}
                        onChange={onUpdateName}/>
                    <label className="input-file">
                        <label style={file || ready ? {} : {color: 'red'}} htmlFor="download">Выберите файл</label>
                        <input id="download" type="file" name="file" ref={inputEl} onChange={onChangeFile}/>
                        <span className="yellow-btn">Открыть</span>
                    </label>
                

                    {
                        drag ?
                            <div id="drop-zone" 
                                onDragStart={e => dragStartHandler(e)}
                                onDragLeave={e => dragLeaveHandler(e)}
                                onDragOver={e => dragStartHandler(e)}
                                onDrop={e => onDropHandler(e)}
                            >
                                <p>Отпустите файл в выделенную область</p>
                            </div>
                            :
                            <div id="drop-zone" 
                                onDragStart={e => dragStartHandler(e)}
                                onDragLeave={e => dragLeaveHandler(e)}
                                onDragOver={e => dragStartHandler(e)}
                            >
                                <p>Переместите файл в выделенную область</p>
                            </div>
                    }

                    <span style ={error ? {color: 'red'}:{}} className="input-file__text">{nameFile}</span>
                    <input className="yellow-btn green-btn" type="submit" value="Начать анализ" />

                </form>
            }
        </div>
    )
}

export default BookDownload;