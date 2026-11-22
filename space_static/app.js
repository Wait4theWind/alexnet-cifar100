const MODEL_URL = "alexnet_cifar100.onnx";
const CLASSES_URL = "classes.json";
const MEAN = [0.5071, 0.4867, 0.4408];
const STD = [0.2675, 0.2565, 0.2761];

const fileInput = document.getElementById("file");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const runButton = document.getElementById("run");
const resultsBox = document.getElementById("results");
const statusBox = document.getElementById("status");

let session = null;
let classes = [];
let currentBitmap = null;
let previewUrl = null;

if (typeof ort === "undefined") {
  statusBox.textContent = "ONNX Runtime 加载失败，请检查网络";
} else {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = 1;
  loadModel();
}

async function loadModel() {
  try {
    const [created, labels] = await Promise.all([
      ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] }),
      fetch(CLASSES_URL).then((response) => response.json()),
    ]);
    session = created;
    classes = labels;
    statusBox.textContent = "模型已就绪，可以上传图片";
    runButton.disabled = !currentBitmap;
  } catch (error) {
    statusBox.textContent = "模型加载失败：" + error.message;
  }
}

function setImage(bitmap) {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  currentBitmap = bitmap;
  previewUrl = URL.createObjectURL(bitmap);
  preview.src = previewUrl;
  preview.hidden = false;
  placeholder.hidden = true;
  runButton.disabled = !session;
  resultsBox.innerHTML = "";
  statusBox.textContent = session ? "图片已选择" : "模型加载中...";
}

async function readFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    return;
  }

  try {
    const bitmap = await createImageBitmap(file);
    setImage(bitmap);
  } catch (error) {
    statusBox.textContent = "图片读取失败：" + error.message;
  }
}

function preprocess(bitmap) {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0, 32, 32);
  const pixels = context.getImageData(0, 0, 32, 32).data;
  const input = new Float32Array(3 * 32 * 32);

  for (let i = 0; i < 32 * 32; i++) {
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    input[i] = (r - MEAN[0]) / STD[0];
    input[32 * 32 + i] = (g - MEAN[1]) / STD[1];
    input[2 * 32 * 32 + i] = (b - MEAN[2]) / STD[2];
  }

  return new ort.Tensor("float32", input, [1, 3, 32, 32]);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

function renderTop5(probabilities) {
  const ranked = Array.from(probabilities, (prob, index) => ({ prob, index }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5);

  resultsBox.innerHTML = "";

  ranked.forEach((item, rank) => {
    const label = classes[item.index].replaceAll("_", " ");
    const percent = item.prob * 100;

    const row = document.createElement("div");
    row.className = "result-row";

    const head = document.createElement("div");
    head.className = "result-line";

    const name = document.createElement("span");
    name.textContent = `${rank + 1}. ${label}`;

    const score = document.createElement("strong");
    score.textContent = percent.toFixed(2) + "%";

    head.appendChild(name);
    head.appendChild(score);

    const track = document.createElement("div");
    track.className = "bar-track";

    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.width = percent.toFixed(2) + "%";

    track.appendChild(bar);
    row.appendChild(head);
    row.appendChild(track);
    resultsBox.appendChild(row);
  });
}

fileInput.addEventListener("change", (event) => {
  readFile(event.target.files[0]);
});

["dragenter", "dragover"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  readFile(event.dataTransfer.files[0]);
});

window.addEventListener("dragover", (event) => {
  event.preventDefault();
});

window.addEventListener("drop", (event) => {
  event.preventDefault();
});

runButton.addEventListener("click", async () => {
  if (!session || !currentBitmap) {
    return;
  }

  runButton.disabled = true;
  statusBox.textContent = "推理中...";

  try {
    const tensor = preprocess(currentBitmap);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const outputs = await session.run({ [inputName]: tensor });
    const probabilities = softmax(outputs[outputName].data);
    renderTop5(probabilities);
    statusBox.textContent = "预测完成";
  } catch (error) {
    statusBox.textContent = "推理失败：" + error.message;
  } finally {
    runButton.disabled = false;
  }
});
