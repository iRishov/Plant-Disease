// Tailwind config
tailwind.config = {
  theme: {
    extend: {
      fontFamily: {
        raleway: ["Raleway", "sans-serif"],
      },
      colors: {
        "agro-green": "#2e7d32",
        "agro-light": "#e8f5e9",
        "agro-dark": "#1b5e20",
      },
    },
  },
};

// DOM Elements
const uploadArea = document.getElementById("uploadArea");
const imageUpload = document.getElementById("imageUpload");
const imagePreview = document.getElementById("imagePreview");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");
const predictBtn = document.getElementById("predictBtn");
const clearImageBtn = document.getElementById("clearImageBtn");
const loadingDiv = document.getElementById("loading");
const resultDiv = document.getElementById("result");

// Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  uploadArea.addEventListener("click", () => imageUpload.click());
  imageUpload.addEventListener("change", uploadImage);
  predictBtn.addEventListener("click", predictImage);
  clearImageBtn.addEventListener("click", clearImage);
  setupDragAndDrop();
});

// Functions
function uploadImage() {
  const file = imageUpload.files[0];

  if (!validateImage(file)) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    imagePreviewContainer.classList.remove("hidden");
    predictBtn.classList.remove("hidden");
    uploadArea.classList.add("hidden");
  };
  reader.readAsDataURL(file);
}

function validateImage(file) {
  if (!file) {
    alert("⚠️ Please select an image.");
    return false;
  }

  const validTypes = ["image/jpeg", "image/png", "image/jpg"];
  if (!validTypes.includes(file.type)) {
    alert("⚠️ Please upload a valid image file (JPEG, JPG, PNG).");
    return false;
  }

  if (file.size > 5 * 1024 * 1024) {
    alert("⚠️ Image size should be less than 5MB.");
    return false;
  }

  return true;
}

function clearImage() {
  imageUpload.value = "";
  imagePreviewContainer.classList.add("hidden");
  predictBtn.classList.add("hidden");
  uploadArea.classList.remove("hidden");
  resultDiv.innerHTML = `
        <div class="text-center text-gray-400">
          <i class="fas fa-seedling text-4xl mb-3"></i>
          <p>No analysis performed yet</p>
          <p class="text-sm mt-2">Upload an image and click "Analyze" to get started</p>
        </div>
      `;
}

async function predictImage() {
  const file = imageUpload.files[0];
  if (!file) {
    alert("⚠️ Please select an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  // Show loading state
  resultDiv.innerHTML = "";
  loadingDiv.classList.remove("hidden");

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    displayResults(data);
  } catch (error) {
    showError(error);
  } finally {
    loadingDiv.classList.add("hidden");
  }
}

function displayResults(data) {
  if (data.error) {
    resultDiv.innerHTML = `
          <div class="bg-red-50 border-l-4 border-red-500 p-4">
            <div class="flex">
              <div class="flex-shrink-0">
                <i class="fas fa-exclamation-circle text-red-500 text-xl"></i>
              </div>
              <div class="ml-3">
                <p class="text-sm text-red-700 font-medium">${data.error}</p>
              </div>
            </div>
          </div>
        `;
  } else {
    resultDiv.innerHTML = `
          <div class="space-y-4">
            <div class="bg-white p-4 rounded-lg shadow-sm">
              <h3 class="font-bold text-lg text-agro-green mb-2">Diagnosis Result</h3>
              <div class="flex items-center justify-between">
                <span class="font-medium">Condition:</span>
                <span class="px-3 py-1 rounded-full text-sm font-semibold ">${
                  data.class
                }</span>
              </div>
              <div class="mt-2 flex items-center justify-between">
                <span class="font-medium">Confidence:</span>
                <div class="w-1/2 bg-gray-200 rounded-full h-2.5">
                  <div class="bg-agro-green h-2.5 rounded-full" 
                       style="width: ${(data.confidence * 100).toFixed(
                         0
                       )}%"></div>
                </div>
                <span class="text-sm font-medium ml-2">${(
                  data.confidence * 100
                ).toFixed(2)}%</span>
              </div>
            </div>

            <div class="bg-white p-4 rounded-lg shadow-sm">
              <h3 class="font-bold text-lg text-agro-green mb-2">Disease Details</h3>
              <div class="space-y-3">
                <div>
                  <h4 class="font-semibold text-gray-800">Symptoms:</h4>
                  <p class="text-gray-600">${
                    data.details.symptoms || "Not specified"
                  }</p>
                </div>
                <div>
                  <h4 class="font-semibold text-gray-800">Disease Cycle:</h4>
                  <p class="text-gray-600">${
                    data.details.disease_cycle || "Not specified"
                  }</p>
                </div>
                <div>
                  <h4 class="font-semibold text-gray-800">Recommended Treatment:</h4>
                  <p class="text-gray-600">${
                    data.details.pesticide_usage || "Not specified"
                  }</p>
                </div>
              </div>
            </div>

         
          </div>
        `;
  }
}

function showError(error) {
  resultDiv.innerHTML = `
        <div class="bg-red-50 border-l-4 border-red-500 p-4">
          <div class="flex">
            <div class="flex-shrink-0">
              <i class="fas fa-exclamation-circle text-red-500 text-xl"></i>
            </div>
            <div class="ml-3">
              <p class="text-sm text-red-700 font-medium">Error: ${error.message}</p>
              <p class="text-xs text-red-600 mt-1">Please try again or contact support</p>
            </div>
          </div>
        </div>
      `;
}

function setupDragAndDrop() {
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    uploadArea.addEventListener(
      eventName,
      () => uploadArea.classList.add("dragover"),
      false
    );
  });

  ["dragleave", "drop"].forEach((eventName) => {
    uploadArea.addEventListener(
      eventName,
      () => uploadArea.classList.remove("dragover"),
      false
    );
  });

  uploadArea.addEventListener("drop", handleDrop, false);

  function handleDrop(e) {
    const files = e.dataTransfer.files;
    if (files.length) {
      imageUpload.files = files;
      uploadImage();
    }
  }
}
