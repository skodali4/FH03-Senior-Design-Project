import React, { useState } from 'react';

function CurrencyGUI() {
  const [usdc, setUsdc] = useState(false);
  const [usdt, setUsdt] = useState(false);
  const [dai, setDai] = useState(false);
  const [output, setOutput] = useState([]);

  const fetchData = async () => {
    let data = [];
    try {
      if (usdc) {
        const response = await fetch('/usdc90days.json');
        if (!response.ok) {
          throw new Error(`Failed to fetch data for USDC: ${response.status} ${response.statusText}`);
        }
        const usdcData = await response.json();
        data.push({ currency: 'USDC', prices: usdcData.prices.map(entry => entry[1]), timestamp:usdcData.prices.map(price => price[0]) });
        console.log('USDC Prices:', usdcData.prices.map(entry => entry[1]));
        console.log('Formatted Dates:', usdcData.prices.map(price => price[0]));
      }
      if (usdt) {
        const response = await fetch('usdt_prices.json');
        if (!response.ok) {
          throw new Error(`Failed to fetch data for USDT: ${response.status} ${response.statusText}`);
        }
        const usdtData = await response.json();
        data.push({ currency: 'USDT', prices: usdtData });
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
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  console.log('OUTPUT:',output); // Log currencyData.prices here

  return (
    <div>
      <h1>Currency GUI</h1>
      <div>
        <label>
          <input type="checkbox" checked={usdc} onChange={() => setUsdc(!usdc)} /> USDC
        </label>
      </div>
      <div>
        <label>
          <input type="checkbox" checked={usdt} onChange={() => setUsdt(!usdt)} /> USDT
        </label>
      </div>
      <div>
        <label>
          <input type="checkbox" checked={dai} onChange={() => setDai(!dai)} /> DAI
        </label>
      </div>
      <div>
        <button onClick={fetchData}>Update</button>
      </div>
      <div>
        {output.map((currencyData, index) => (
          <div key={index}>
            <h3>{currencyData.currency}</h3>
            <ul>
              {currencyData.prices.map((price, idx) => (
              <li key={idx}>
                {currencyData.timestamp[idx]}: {price}
              </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

export default CurrencyGUI;