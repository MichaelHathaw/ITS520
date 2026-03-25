// 0 = L
// 1 = T

const TRAINING_DATA = [
  // =========================
  // L samples (60)
  // =========================
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 0,1,1,0], y:0 },

  { x:[1,0,0,0, 1,0,0,0, 1,0,0,1, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 0,0,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 0,1,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 0,0,0,0, 1,0,0,0, 1,1,1,0], y:0 },

  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,1], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,1], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 1,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,1], y:0 },

  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 0,0,0,0, 1,1,1,0], y:0 },

  { x:[1,0,0,0, 1,0,0,0, 1,0,0,1, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 0,1,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 0,0,0,0, 1,0,0,0, 1,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,1], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:0 },

  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 1,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,1, 0,1,1,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,1], y:0 },

  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,1, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 0,0,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,1], y:0 },

  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 1,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,1, 0,1,0,0, 0,1,1,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,0], y:0 },

  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,1, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 0,0,0,0, 1,0,0,0, 1,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,1], y:0 },

  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 1,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,1, 1,1,1,0], y:0 },

  { x:[1,0,0,0, 1,0,0,0, 0,0,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,1], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 1,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:0 },

  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,0], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,1], y:0 },
  { x:[1,0,0,0, 1,0,0,0, 0,1,0,0, 1,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,0, 1,1,1,0], y:0 },
  { x:[0,1,0,0, 0,1,0,0, 0,1,0,1, 0,1,1,1], y:0 },

  // =========================
  // T samples (60)
  // =========================
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,1, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,1, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 0,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,1, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,1, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 0,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[1,1,1,0, 0,1,0,0, 0,1,0,1, 0,1,0,0], y:1 },
  { x:[1,1,1,0, 0,1,0,1, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,1, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,1, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 0,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,1, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,1, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 0,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,1, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,0, 0,1,0,1, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,1, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,1, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,1], y:1 },

  { x:[1,1,1,0, 0,1,0,1, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[1,1,1,1, 0,1,0,0, 0,1,0,0, 0,1,0,0], y:1 },
  { x:[0,1,0,0, 1,1,1,0, 0,1,0,0, 0,1,1,0], y:1 },
  { x:[1,1,1,0, 0,1,0,0, 0,1,0,0, 0,1,0,1], y:1 }
];

// -----------------------------
// Perceptron
// -----------------------------
class Perceptron {
  constructor(inputSize, learningRate = 0.1, epochs = 25) {
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

// -----------------------------
// Prepare training arrays
// -----------------------------
const X_train = TRAINING_DATA.map(item => item.x);
const y_train = TRAINING_DATA.map(item => item.y);

const perceptron = new Perceptron(16, 0.1, 30);
perceptron.train(X_train, y_train);

// -----------------------------
// UI setup
// -----------------------------
const gridElement = document.getElementById("grid");
const predictionElement = document.getElementById("prediction");
const rawOutputElement = document.getElementById("rawOutput");
const accuracyElement = document.getElementById("accuracy");
const vectorDisplay = document.getElementById("vectorDisplay");

let currentGrid = new Array(16).fill(0);

function updateAccuracyDisplay() {
  const acc = perceptron.accuracy(X_train, y_train).toFixed(2);
  accuracyElement.textContent = `${acc}%`;
}

function updateVectorDisplay() {
  vectorDisplay.textContent = `[${currentGrid.join(",")}]`;
}

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

function clearGrid() {
  currentGrid = new Array(16).fill(0);
  predictionElement.textContent = "None";
  rawOutputElement.textContent = "0";
  renderGrid();
  updateVectorDisplay();
}

function classifyGrid() {
  const result = perceptron.predict(currentGrid);
  predictionElement.textContent = result.pred === 0 ? "L" : "T";
  rawOutputElement.textContent = result.raw.toFixed(3);
}

function randomSample() {
  const randomIndex = Math.floor(Math.random() * TRAINING_DATA.length);
  currentGrid = [...TRAINING_DATA[randomIndex].x];
  renderGrid();
  updateVectorDisplay();
  classifyGrid();
}

function retrainModel() {
  perceptron.weights = new Array(16).fill(0);
  perceptron.bias = 0;
  perceptron.train(X_train, y_train);
  updateAccuracyDisplay();
  alert("Perceptron retrained on training dataset.");
}

// Buttons
document.getElementById("classifyBtn").addEventListener("click", classifyGrid);
document.getElementById("clearBtn").addEventListener("click", clearGrid);
document.getElementById("randomBtn").addEventListener("click", randomSample);
document.getElementById("trainBtn").addEventListener("click", retrainModel);

// Initial render
renderGrid();
updateVectorDisplay();
updateAccuracyDisplay();
