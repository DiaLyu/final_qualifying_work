import './App.scss';
import MapPage from '../pages/MapPage';
import DescrPage from '../pages/DescrPage';
import Page404 from '../pages/404';

import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

const App = () => {

  return(
    <Router>
      <Routes>
        <Route path='/' element={<DescrPage/>}/>
        <Route path='/map' element={<MapPage/>}/>
        <Route path='*' element={<Page404/>}/>
      </Routes>
    </Router>
  );
}

export default App;
