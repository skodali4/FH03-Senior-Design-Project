import React, { useState } from 'react';

function CurrencyGUI() {
  const [usdc, setUsdc] = useState(false);
  const [usdt, setUsdt] = useState(false);
  const [dai, setDai] = useState(false);
  const [output, setOutput] = useState([]);

  const updateOutput = () => {
    setOutput([...output, `USDC: ${usdc}, USDT: ${usdt}, DAI: ${dai}`]);
  };

  return (
    <div>
      <h1>Currency GUI</h1>
      <label>
        <input type="checkbox" checked={usdc} onChange={() => setUsdc(!usdc)} /> USDC
      </label>
      <label>
        <input type="checkbox" checked={usdt} onChange={() => setUsdt(!usdt)} /> USDT
      </label>
      <label>
        <input type="checkbox" checked={dai} onChange={() => setDai(!dai)} /> DAI
      </label>
      <button onClick={updateOutput}>Update</button>
      <textarea readOnly value={output.join('\n')} rows={5} cols={50} />
    </div>
  );
}

export default CurrencyGUI;