class Perceptron {
  constructor(inputSize, learningRate = 0.1, epochs = 40) {
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
        const out = this.predict(X[i]).pred;
        const error = y[i] - out;

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

let perceptron = new Perceptron(16);
let TRAINING_DATA = [];
let currentGrid = new Array(16).fill(0);

const gridElement = document.getElementById("grid");
const predictionElement = document.getElementById("prediction");
const rawOutputElement = document.getElementById("rawOutput");
const accuracyElement = document.getElementById("accuracy");
const vectorDisplay = document.getElementById("vectorDisplay");
const totalSamplesElement = document.getElementById("totalSamples");

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
  rawOutputElement.textContent = result.raw.toFixed(3);
}

function showAccuracy() {
  const X = TRAINING_DATA.map(item => item.x);
  const y = TRAINING_DATA.map(item => item.y);
  const acc = perceptron.accuracy(X, y).toFixed(2);
  accuracyElement.textContent = `${acc}%`;
}

function retrain() {
  perceptron = new Perceptron(16);
  const X = TRAINING_DATA.map(item => item.x);
  const y = TRAINING_DATA.map(item => item.y);
  perceptron.train(X, y);
  showAccuracy();
  alert("Model retrained.");
}

function randomSample() {
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

  perceptron.train(X, y);
  showAccuracy();

  totalSamplesElement.textContent = TRAINING_DATA.length;
}

document.getElementById("classifyBtn").addEventListener("click", classifyGrid);
document.getElementById("clearBtn").addEventListener("click", clearGrid);
document.getElementById("randomBtn").addEventListener("click", randomSample);
document.getElementById("trainBtn").addEventListener("click", retrain);

renderGrid();
updateVectorDisplay();
loadDataset();
