import './App.css';
import Navabar from './components/navbar';
import Header from './components/header';
import ChatUi from './components/chatUi';
import RNN from './components/rnn'
import CBOW from './components/cbow'
import { useState } from 'react';

function App() {
  const [route, setRoute] = useState('cbow')
  
  return (
    <>
      {/* <Navabar /> */}
      <Header route={route} setRoute={setRoute}/>
      {route === 'cbow' ?
       <>
      <RNN/>
      <CBOW/>
      </>
      :
      <ChatUi />}
    </>
  );
}

export default App;
