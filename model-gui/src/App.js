import React, { useState } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import { Line } from 'react-chartjs-2';
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns';

const downsampleData = (data, maxDataPoints) => {
  const downsampledData = [];
  const interval = Math.ceil(data.length / maxDataPoints);
  for (let i = 0; i < data.length; i += interval) {
    downsampledData.push(data[i]);
  }
  return downsampledData;
};
const maxDataPoints = 300;
function App() {
    const [currency, setCurrency] = useState('');
    const [usdc, setUsdc] = useState(false);
    const [usdt, setUsdt] = useState(false);
    const [eth, setEth] = useState(false);
    const [daimult, setDaiM] = useState(false);
    const [daiuni, setDaiU] = useState(false);
    const [output, setOutput] = useState([]);
    const [simOutput, setsimOutput] = useState([]);
    const [showChart, setShowChart] = useState(false);
    const [formData, setFormData] = useState({
      num_sims: '',
      alpha: '',
      beta: '',
      num_ethereum: '',
      num_stablecoin: ''
    });

    const numSims = parseInt(formData.num_sims);
    const alpha = parseFloat(formData.alpha);
    const beta = parseFloat(formData.beta);
    const numEthereum = parseInt(formData.num_ethereum);
    const numStablecoin = parseInt(formData.num_stablecoin);    

    const handleSubmit = async() => {
      // Submit form data to backend Flask application
      try {
        const response = await fetch(`http://localhost:5000/simulation?num_sims=${numSims}&alpha=${alpha}&beta=${beta}&num_ethereum=${numEthereum}&num_stablecoin=${numStablecoin}`, {
            method: 'GET'
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setsimOutput(data);
        console.log('Response:', data);
    } catch (error) {
        console.error('Error:', error);
    }
  
    };

    const handleChange = (e) => {
      const { name, value } = e.target;
      setFormData({ ...formData, [name]: value });
    };

    const fetchData = async () => {
      let data = [];
      try {
        if (usdc) {
          const response = await fetch('http://localhost:5000/usdc');
          if (!response.ok) {
            throw new Error(`Failed to fetch data for USDC: ${response.status} ${response.statusText}`);
          }
          const usdcData = await response.json();
          console.log('Fetched USDC data:', usdcData); // Log fetched data
            data.push({
              currency: 'USDC',
              prices: usdcData.map(entry => entry['predicted price'][0]),
              timestamp: usdcData.map(price => price['timestamp'])
            });
            console.log('USDC Prices:', usdcData.map(entry => entry['predicted price'][0]));
            console.log('Formatted Dates:', usdcData.map(price => price['timestamp']));
        }
        if (usdt) {
          const response = await fetch('http://localhost:5000/usdt');
          if (!response.ok) {
            throw new Error(`Failed to fetch data for USDT: ${response.status} ${response.statusText}`);
          }
          const usdtData = await response.json();
          console.log('Fetched USDC data:', usdtData); // Log fetched data
            data.push({
              currency: 'USDT',
              prices: usdtData.map(entry => entry['predicted price'][0]),
              timestamp: usdtData.map(price => price['timestamp'])
            });
            console.log('USDC Prices:', usdtData.map(entry => entry['predicted price'][0]));
            console.log('Formatted Dates:', usdtData.map(price => price['timestamp']));
        }
        if (eth) {
          const response = await fetch('http://localhost:5000/eth');
          if (!response.ok) {
            throw new Error(`Failed to fetch data for USDC: ${response.status} ${response.statusText}`);
          }
          const ethData = await response.json();
          console.log('Fetched ETH data:', ethData); // Log fetched data
            data.push({
              currency: 'ETH',
              prices: ethData.map(entry => entry['predicted price'][0]),
              timestamp: ethData.map(price => price['timestamp'])
            });
            console.log('USDC Prices:', ethData.map(entry => entry['predicted price'][0]));
            console.log('Formatted Dates:', ethData.map(price => price['timestamp']));
        }
        if (daimult) {
          const response = await fetch('http://localhost:5000/dai/multi');
          if (!response.ok) {
            throw new Error(`Failed to fetch data for DAI Multi: ${response.status} ${response.statusText}`);
          }
          const daiMData = await response.json();
          console.log('Fetched DAI Multi data:', daiMData); // Log fetched data
            data.push({
              currency: 'DAI Multi Input',
              prices: daiMData.map(entry => entry['predicted price'][0]),
              timestamp: daiMData.map(price => price['timestamp'])
            });
            console.log('DAI Multi Input Prices:', daiMData.map(entry => entry['predicted price'][0]));
            console.log('Formatted Dates:', daiMData.map(price => price['timestamp']));
        }
        if (daiuni) {
          const response = await fetch('http://localhost:5000/dai/uni');
          if (!response.ok) {
            throw new Error(`Failed to fetch data for DAI uni: ${response.status} ${response.statusText}`);
          }
          const daiUData = await response.json();
          console.log('Fetched DAI uni data:', daiUData); // Log fetched data
            data.push({
              currency: 'DAI Uni Input',
              prices: daiUData.map(entry => entry['predicted price'][0]),
              timestamp: daiUData.map(price => price['timestamp'])
            });
            console.log('DAI Multi Input Prices:', daiUData.map(entry => entry['predicted price'][0]));
            console.log('Formatted Dates:', daiUData.map(price => price['timestamp']));
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

  const chartData2 = {
    labels: simOutput.length > 0 ? Array.from({ length: simOutput.length }, (_, i) => i + 1) : [],
    datasets: simOutput.length > 0 ? [
      {
        label: 'Price',
        data: simOutput,
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
    <div className="App">
      <Tabs>
        <TabList>
          <Tab>Time Series Forecasting</Tab>
          <Tab>Simulation</Tab>
        </TabList>

        <TabPanel>
          <div className="app-container">
            <h1>Currency GUI</h1>
            <div className="input-container">
            <label>
              <input type="radio" name="currency" value="USDC" onChange={(e) => { setUsdc(true); setUsdt(false); setEth(false); setDaiM(false); setDaiU(false);}} /> USDC
            </label>
            <label>
              <input type="radio" name="currency" value="USDT" onChange={(e) => { setUsdc(false); setUsdt(true); setEth(false); setDaiM(false); setDaiU(false)}} /> USDT
            </label>
            <label>
              <input type="radio" name="currency" value="ETH" onChange={(e) => { setUsdc(false); setUsdt(false); setEth(true); setDaiM(false); setDaiU(false)}} /> ETH
            </label>
            <label>
              <input type="radio" name="currency" value="DAI MULTI" onChange={(e) => { setUsdc(false); setUsdt(false); setEth(false); setDaiM(true); setDaiU(false)}} /> DAI(multi)
            </label>
            <label>
              <input type="radio" name="currency" value="DAI UNI" onChange={(e) => { setUsdc(false); setUsdt(false); setEth(false); setDaiM(false); setDaiU(true)}} /> DAI(uni)
            </label>
            </div>
              <button onClick={handleButtonClick}>Update</button>
            <div>
            </div>
            {(
            <div>
            <h2>My Chart</h2>
            <div className="chart-container" style={{ width: '1000px', height: '10000' }}></div>
            <Line data={chartData} options={options}/>
          </div>
          )}
          </div>
        </TabPanel>
        <TabPanel>
        <div>
        <label>Number of Simulations:</label>
        <input type="text" name="num_sims" value={formData.num_sims} onChange={handleChange} />
      </div>
      <div>
        <label>Field 2:</label>
        <input type="text" name="alpha" value={formData.alpha} onChange={handleChange} />
      </div>
      <div>
        <label>Field 3:</label>
        <input type="text" name="beta" value={formData.beta} onChange={handleChange} />
      </div>
      <div>
        <label>Field 4:</label>
        <input type="text" name="num_ethereum" value={formData.num_ethereum} onChange={handleChange} />
      </div>
      <div>
        <label>Field 5:</label>
        <input type="text" name="num_stablecoin" value={formData.num_stablecoin} onChange={handleChange} />
      </div>
      <button onClick={handleSubmit}>Submit</button>
            {(
            <div>
            <h2>My Chart</h2>
            <div className="chart-container" style={{ width: '1000px', height: '10000' }}></div>
            <Line data={chartData2} options={options}/>
          </div>
          )}
        </TabPanel>
      </Tabs>
    </div>
  );
}

export default App;