import React, { useState } from 'react';
import './App.css'; // Import your CSS file

function App() {
  const [currency, setCurrency] = useState('');
  const [startDate, setStartDate] = useState('');
  const [startTime, setStartTime] = useState('');
  const [endDate, setEndDate] = useState('');
  const [endTime, setEndTime] = useState('');
  const [output, setOutput] = useState([]);

  const updateOutput = () => {
    setOutput([...output, `Currency: ${currency}, Start Date: ${startDate}, Start Time: ${startTime}, End Date: ${endDate}, End Time: ${endTime}`]);
  };

  return (
    <div className="app-container">
      <h1>Currency GUI</h1>
      <div className="input-container">
        <label>
          <input type="radio" name="currency" value="USDC" onChange={(e) => setCurrency(e.target.value)} /> USDC
        </label>
        <label>
          <input type="radio" name="currency" value="USDT" onChange={(e) => setCurrency(e.target.value)} /> USDT
        </label>
        <label>
          <input type="radio" name="currency" value="DAI" onChange={(e) => setCurrency(e.target.value)} /> DAI
        </label>
        <h2>Start Time</h2>
        <input type="date" onChange={(e) => setStartDate(e.target.value)} />
        <input type="time" onChange={(e) => setStartTime(e.target.value)} />
        <h2>End Time</h2>
        <input type="date" onChange={(e) => setEndDate(e.target.value)} />
        <input type="time" onChange={(e) => setEndTime(e.target.value)} />
        <button onClick={updateOutput}>Update</button>
      </div>
      <div className="output-container">
        <textarea readOnly value={output.join('\n')} rows={5} cols={50} />
      </div>
    </div>
  );
}

export default App;