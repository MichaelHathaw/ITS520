let meta, session;
async function load() {
  meta = await (await fetch('preprocess_cls.json')).json();
  session = await ort.InferenceSession.create('adult_cls.onnx');
  const f = document.getElementById('form');
  meta.num_cols.forEach(n=>{
    const d = document.createElement('div');
    d.innerHTML = `<label>${n}: <input type="number" step="any" id="num_${n}"></label>`;
    f.appendChild(d);
  });
  meta.cat_cols.forEach((n, i)=>{
    const opts = meta.cat_categories[i];
    const d = document.createElement('div');
    d.innerHTML = `<label>${n}: <select id="cat_${n}">${opts.map(o=>`<option>${o}</option>`).join('')}</select></label>`;
    f.appendChild(d);
  });
}
function buildFeature() {
  const num = meta.num_cols.map((n,i)=>{
    const v = Number(document.getElementById(`num_${n}`).value||0);
    return (v - meta.num_mean[i]) / meta.num_scale[i];
  });
  const catOH = [];
  meta.cat_cols.forEach((n, i)=>{
    const val = document.getElementById(`cat_${n}`).value;
    const cats = meta.cat_categories[i];
    cats.forEach(c => catOH.push(c === val ? 1.0 : 0.0));
  });
  const x = Float32Array.from([...num, ...catOH]);
  return new ort.Tensor('float32', x, [1, x.length]);
}
async function predict() {
  const input = buildFeature();
  const out = await session.run({input});
  const logits = Array.from(out.logits.data);
  const probs = softmax(logits);
  const pred = probs[1] >= 0.5 ? 1 : 0;
  document.getElementById('out').textContent =
    `P(>50K)=${probs[1].toFixed(3)} -> predicted class: ${pred ? '>50K' : '<=50K'}`;
}
function softmax(arr){
  const m = Math.max(...arr);
  const exps = arr.map(v=>Math.exp(v - m));
  const s = exps.reduce((a,b)=>a+b,0);
  return exps.map(v=>v/s);
}
document.getElementById('btn').onclick = predict;
load();