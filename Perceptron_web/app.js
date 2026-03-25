class Perceptron {
  constructor(inputSize, learningRate = 0.1, epochs = 50) {
    this.inputSize = inputSize;
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.weights = new Array(inputSize).fill(0);
    this.bias = 0;
  }

  activation(z) {
    return z >= 0 ? 1 : 0;
  }

  predict(x) {
    let z = this.bias;
    for (let i = 0; i < this.inputSize; i++) {
      z += this.weights[i] * x[i];
    }
    return {
      raw: z,
      pred: this.activation(z)
    };
  }

  train(X, y) {
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      for (let i = 0; i < X.length; i++) {
        const output = this.predict(X[i]).pred;
        const error = y[i] - output;

        for (let j = 0; j < this.inputSize; j++) {
          this.weights[j] += this.learningRate * error * X[i][j];
        }

        this.bias += this.learningRate * error;
      }
    }
  }

  accuracy(X, y) {
    let correct = 0;
    for (let i = 0; i < X.length; i++) {
      if (this.predict(X[i]).pred === y[i]) {
        correct++;
      }
    }
    return (correct / X.length) * 100;
  }
}

let perceptron = new Perceptron(16, 0.1, 50);
let TRAINING_DATA = [];
let currentGrid = new Array(16).fill(0);

const gridElement = document.getElementById("grid");
const predictionElement = document.getElementById("prediction");
const rawOutputElement = document.getElementById("rawOutput");
const accuracyElement = document.getElementById("accuracy");
const totalSamplesElement = document.getElementById("totalSamples");
const lCountElement = document.getElementById("lCount");
const tCountElement = document.getElementById("tCount");
const vectorDisplay = document.getElementById("vectorDisplay");

function renderGrid() {
  gridElement.innerHTML = "";

  currentGrid.forEach((value, index) => {
    const cell = document.createElement("div");
    cell.className = "cell";

    if (value === 1) {
      cell.classList.add("active");
    }

    cell.addEventListener("click", () => {
      currentGrid[index] = currentGrid[index] === 1 ? 0 : 1;
      renderGrid();
      updateVectorDisplay();
    });

    gridElement.appendChild(cell);
  });
}

function updateVectorDisplay() {
  vectorDisplay.textContent = `[${currentGrid.join(",")}]`;
}

function clearGrid() {
  currentGrid = new Array(16).fill(0);
  renderGrid();
  updateVectorDisplay();
  predictionElement.textContent = "None";
  rawOutputElement.textContent = "0";
}

function classifyGrid() {
  const result = perceptron.predict(currentGrid);
  predictionElement.textContent = result.pred === 0 ? "L" : "T";
  rawOutputElement.textContent = result.raw.toFixed(4);
}

function retrainModel() {
  const X = TRAINING_DATA.map(item => item.x);
  const y = TRAINING_DATA.map(item => item.y);

  perceptron = new Perceptron(16, 0.1, 50);
  perceptron.train(X, y);

  const acc = perceptron.accuracy(X, y).toFixed(2);
  accuracyElement.textContent = `${acc}%`;
  alert("Perceptron retrained.");
}

function loadRandomSample() {
  if (TRAINING_DATA.length === 0) return;

  const randomIndex = Math.floor(Math.random() * TRAINING_DATA.length);
  currentGrid = [...TRAINING_DATA[randomIndex].x];
  renderGrid();
  updateVectorDisplay();
  classifyGrid();
}

async function loadDataset() {
  const response = await fetch("dataset.json");
  TRAINING_DATA = await response.json();

  const X = TRAINING_DATA.map(item => item.x);
  const y = TRAINING_DATA.map(item => item.y);

  const lCount = TRAINING_DATA.filter(item => item.y === 0).length;
  const tCount = TRAINING_DATA.filter(item => item.y === 1).length;

  totalSamplesElement.textContent = TRAINING_DATA.length;
  lCountElement.textContent = lCount;
  tCountElement.textContent = tCount;

  perceptron.train(X, y);

  const acc = perceptron.accuracy(X, y).toFixed(2);
  accuracyElement.textContent = `${acc}%`;
}

document.getElementById("classifyBtn").addEventListener("click", classifyGrid);
document.getElementById("clearBtn").addEventListener("click", clearGrid);
document.getElementById("randomBtn").addEventListener("click", loadRandomSample);
document.getElementById("trainBtn").addEventListener("click", retrainModel);

renderGrid();
updateVectorDisplay();
loadDataset();
