let session = null;
let preprocess = null;

const modelPath = "breast_cancer_cls.onnx";
const preprocessPath = "preprocess_cls.json";

const modelStatus = document.getElementById("modelStatus");
const featureGrid = document.getElementById("featureGrid");
const predictBtn = document.getElementById("predictBtn");
const fillExampleBtn = document.getElementById("fillExampleBtn");
const clearBtn = document.getElementById("clearBtn");
const resultBox = document.getElementById("result");

function setStatus(message, className) {
  modelStatus.textContent = message;
  modelStatus.className = `status ${className}`;
}

function softmax(values) {
  const maxVal = Math.max(...values);
  const exps = values.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function createInputs(meta) {
  featureGrid.innerHTML = "";

  meta.num_cols.forEach((name, index) => {
    const wrapper = document.createElement("div");

    const label = document.createElement("label");
    label.setAttribute("for", `f_${index}`);
    label.textContent = name;

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.id = `f_${index}`;
    input.dataset.index = String(index);
    input.placeholder = `Enter ${name}`;

    wrapper.appendChild(label);
    wrapper.appendChild(input);
    featureGrid.appendChild(wrapper);
  });
}

function getRawInputValues() {
  return preprocess.num_cols.map((_, index) => {
    const value = document.getElementById(`f_${index}`).value.trim();
    return value;
  });
}

function validateAndStandardize() {
  const rawValues = getRawInputValues();
  const standardized = [];

  for (let i = 0; i < rawValues.length; i++) {
    if (rawValues[i] === "") {
      throw new Error(`Missing value for "${preprocess.num_cols[i]}"`);
    }

    const parsed = Number(rawValues[i]);
    if (!Number.isFinite(parsed)) {
      throw new Error(`Invalid numeric value for "${preprocess.num_cols[i]}"`);
    }

    const mean = preprocess.num_mean[i];
    const scale = preprocess.num_scale[i];
    standardized.push((parsed - mean) / scale);
  }

  return standardized;
}

function showPrediction(label, confidence, probs) {
  const malignantPct = (probs[0] * 100).toFixed(2);
  const benignPct = (probs[1] * 100).toFixed(2);

  resultBox.innerHTML = `
    <div class="prediction">Prediction: ${label}</div>
    <div class="confidence">Confidence: ${(confidence * 100).toFixed(2)}%</div>
    <ul class="prob-list">
      <li>Malignant: ${malignantPct}%</li>
      <li>Benign: ${benignPct}%</li>
    </ul>
  `;
}

function showError(message) {
  resultBox.innerHTML = `
    <div class="prediction" style="color:#b91c1c;">Prediction Error</div>
    <div class="confidence">${message}</div>
  `;
}

async function predict() {
  try {
    const inputData = validateAndStandardize();

    const inputTensor = new ort.Tensor(
      "float32",
      Float32Array.from(inputData),
      [1, inputData.length]
    );

    const feeds = { input: inputTensor };
    const output = await session.run(feeds);

    const logits = output.logits.data;
    const probs = softmax(Array.from(logits));

    let maxIndex = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIndex]) maxIndex = i;
    }

    const label = preprocess.target_names[maxIndex];
    const confidence = probs[maxIndex];

    showPrediction(label, confidence, probs);

  } catch (error) {
    showError(error.message || "Unknown prediction error.");
  }
}

function fillExampleValues() {
  preprocess.num_mean.forEach((value, index) => {
    document.getElementById(`f_${index}`).value = String(value);
  });
}

function clearInputs() {
  preprocess.num_cols.forEach((_, index) => {
    document.getElementById(`f_${index}`).value = "";
  });

  resultBox.innerHTML = `
    <div class="prediction">No prediction yet.</div>
    <div class="confidence">Load the model, enter values, and click Predict.</div>
  `;
}

async function init() {
  try {
    setStatus("Loading preprocessing metadata...", "loading");

    const preprocessResp = await fetch(preprocessPath);
    if (!preprocessResp.ok) {
      throw new Error(`Could not load ${preprocessPath}`);
    }
    preprocess = await preprocessResp.json();

    createInputs(preprocess);

    setStatus("Loading ONNX model...", "loading");

    session = await ort.InferenceSession.create(modelPath);

    setStatus("Model loaded successfully.", "ready");

    predictBtn.disabled = false;
    fillExampleBtn.disabled = false;
    clearBtn.disabled = false;

  } catch (error) {
    console.error(error);
    setStatus(`Failed to load app files: ${error.message}`, "error");
    showError(error.message || "Initialization failed.");
  }
}

predictBtn.addEventListener("click", predict);
fillExampleBtn.addEventListener("click", fillExampleValues);
clearBtn.addEventListener("click", clearInputs);

init();
