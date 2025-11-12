let meta, session;
async function load() {
  meta = await (await fetch('preprocess_reg.json')).json();
  session = await ort.InferenceSession.create('california_regression.onnx');
  const f = document.getElementById('form');
  meta.feature_names.forEach(name=>{
    const d = document.createElement('div');
    d.innerHTML = `<label>${name}: <input type="number" step="any" id="f_${name}"></label>`;
    f.appendChild(d);
  });
}
function standardize(vals) {
  const mu = meta.scaler_mean, sc = meta.scaler_scale;
  return vals.map((v,i)=> (v - mu[i]) / sc[i]);
}
async function predict() {
  const vals = meta.feature_names.map(n => Number(document.getElementById(`f_${n}`).value||0));
  const x = standardize(vals);
  const input = new ort.Tensor('float32', Float32Array.from(x), [1, x.length]);
  const out = await session.run({input});
  const y = out.pred.data[0];
  document.getElementById('out').textContent = `Predicted median house value: ${y.toFixed(3)} (x $100k)`;
}
document.getElementById('btn').onclick = predict;
load();