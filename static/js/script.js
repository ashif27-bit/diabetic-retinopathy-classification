const input = document.getElementById("image");
const preview = document.getElementById("preview");
const dropzone = document.getElementById("dropzone");
const predictBtn = document.getElementById("predictBtn");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent = document.getElementById("resultContent");
const prediction = document.getElementById("prediction");
const description = document.getElementById("description");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const resultImage = document.getElementById("resultImage");
const riskPill = document.getElementById("riskPill");
const demoWarning = document.getElementById("demoWarning");
const statusDot = document.getElementById("statusDot");

let selectedFile = null;

function showPreview(file) {
  if (!file) return;
  const valid = ["image/jpeg", "image/png"];
  if (!valid.includes(file.type)) {
    alert("Please choose a JPG, JPEG or PNG image.");
    return;
  }

  selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.innerHTML = `<img src="${url}" alt="Retinal image preview">`;
  preview.classList.add("has-image");
  predictBtn.disabled = false;
  statusDot.classList.add("ready");
}

input.addEventListener("change", () => showPreview(input.files[0]));

["dragenter", "dragover"].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  });
});
["dragleave", "drop"].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  showPreview(file);
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  predictBtn.innerHTML = "<span>Analyzing...</span><span>⏳</span>";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) throw new Error(data.error || "Prediction failed.");

    resultPlaceholder.classList.add("hidden");
    resultContent.classList.remove("hidden");

    prediction.textContent = data.prediction;
    description.textContent = data.description;
    confidenceText.textContent = `${data.confidence}%`;
    confidenceBar.style.width = `${Math.min(100, data.confidence)}%`;
    resultImage.src = data.image_url;

    riskPill.textContent = data.risk.toUpperCase();
    riskPill.style.background = data.risk === "Low" ? "#eaf8f3" : "#fff1e9";
    riskPill.style.color = data.risk === "Low" ? "#16875f" : "#c75d1a";

    demoWarning.style.display = data.demo_model ? "block" : "none";
  } catch (err) {
    alert(err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = "<span>Analyze Image</span><span>→</span>";
  }
});
