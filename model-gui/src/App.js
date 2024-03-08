import React, { useState } from 'react';
import './App.css'; // Import your CSS file
import { Line } from 'react-chartjs-2';
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns';
// Function to aggregate data into intervals and calculate aggregate values

// Function to downsample data to limit the number of data points
const downsampleData = (data, maxDataPoints) => {
  const downsampledData = [];
  const interval = Math.ceil(data.length / maxDataPoints);
  for (let i = 0; i < data.length; i += interval) {
    downsampledData.push(data[i]);
  }
  return downsampledData;
};

const maxDataPoints = 300; // Adjust as needed
function App() {
  const [currency, setCurrency] = useState('');
  const [usdc, setUsdc] = useState(false);
  const [usdt, setUsdt] = useState(false);
  const [dai, setDai] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [startTime, setStartTime] = useState('');
  const [endDate, setEndDate] = useState('');
  const [endTime, setEndTime] = useState('');
  const [output, setOutput] = useState([]);
  const [showChart, setShowChart] = useState(false);



  const fetchData = async () => {
    let data = [];
    try {
      if (usdc) {
        const response = await fetch('/usdc90days.json');
        if (!response.ok) {
          throw new Error(`Failed to fetch data for USDC: ${response.status} ${response.statusText}`);
        }
        const usdcData = await response.json();
        console.log('Fetched USDC data:', usdcData); // Log fetched data

        if (usdcData && Array.isArray(usdcData.prices)) {
          if (!usdcData || !Array.isArray(usdcData.prices)) {
            throw new Error('Invalid data format for USDC');
          }
          const downsampledData = downsampleData(usdcData.prices, maxDataPoints);
          console.log('new shorter data:', downsampledData);
          data.push({
            currency: 'USDC',
            prices: downsampledData.map(entry => entry[1]),
            timestamp: downsampledData.map(entry => entry[0])
          });
          console.log('USDC Prices:', downsampledData.map(entry => entry[1]));
          console.log('Formatted Dates:', downsampledData.map(price => price[0]));
        }
        else {
          console.error('Invalid format for USDC data:', usdcData);
        }
      }
      if (usdt) {
        const response = await fetch('/ethereum90days.json');
        if (!response.ok) {
          throw new Error(`Failed to fetch data for USDT: ${response.status} ${response.statusText}`);
        }
        const usdtData = await response.json();
        const downsampledData = downsampleData(usdtData.prices, maxDataPoints);
        console.log('new shorter data:', downsampledData);
        data.push({
          currency: 'USDT',
          prices: downsampledData.map(entry => entry[1]),
          timestamp: downsampledData.map(entry => entry[0])
        });
        console.log('USDT Prices:', downsampledData.map(entry => entry[1]));
        console.log('Formatted Dates:', downsampledData.map(price => price[0]));
  
      }
      if (dai) {
        const response = await fetch('dai_prices.json');
        if (!response.ok) {
          throw new Error(`Failed to fetch data for DAI: ${response.status} ${response.statusText}`);
        }
        const daiData = await response.json();
        data.push({ currency: 'DAI', prices: daiData });
      }
      setOutput(data);
      console.log('OUTPUT:',data); // Log currencyData.prices here
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleButtonClick = () => {
    setShowChart(true);
    fetchData();
  };

console.log('OUTPUT:', output); // Log output after setting state

// Check the structure of output in the render method
console.log('OUTPUT Length:', output.length);
if (output.length > 0) {
  console.log('First Output:', output[0].prices);
  console.log('First Output:', output[0].timestamp);
}

const exchartData = {
  labels: [1701295258512, 1702850480130, 1704405648257, 1705964536797, 1707519660513],
  datasets: [
    {
      label: 'My Dataset',
      data: [1.0003, 1.0001, 0.9995, 1.001, 1.0004],
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    }
  ]
};


const chartData = {
  labels: output.length > 0 ? output[0].timestamp.map(value => parseFloat(value)) : [],
  datasets: output.length > 0 ? [
    {
      label: output[0].currency,
      data: output[0].prices.map(value => parseFloat(value)),
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    }
  ] : []
};
  

  const options = {
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Value',
        },
      },
    },
  };


  return (
    <div className="app-container">
      <h1>Currency GUI</h1>
      <div className="input-container">
        <label>
          <input type="radio" name="currency" value="USDC" onChange={(e) => setUsdc(!usdc)} /> USDC
        </label>
        <label>
          <input type="radio" name="currency" value="USDT" onChange={(e) => setUsdt(!usdt)} /> USDT
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
        <button onClick={handleButtonClick}>Update</button>
      </div>
      {showChart && (
      <div>
      <h2>My Chart</h2>
      <div className="chart-container" style={{ width: '1000px', height: '10000' }}></div>
      <Line data={chartData} options={options}/>
    </div>
    )}
    </div>
  );
}

export default App;