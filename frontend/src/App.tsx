import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { App as AntdApp } from 'antd';
import Home from './pages/home';
import './styles/global.css';
import './App.css';

function App() {
  return (
    <AntdApp className="App">
      <Router>
        <Routes>
          <Route path="/" element={<Home/>} />
        </Routes>
      </Router>
    </AntdApp>
  );
}

export default App;