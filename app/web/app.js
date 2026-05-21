const state = {
  imageA: null,
  imageB: null,
};

const el = {
  fileA: document.getElementById("file-a"),
  fileB: document.getElementById("file-b"),
  dropzoneA: document.getElementById("dropzone-a"),
  dropzoneB: document.getElementById("dropzone-b"),
  previewA: document.getElementById("preview-a"),
  previewB: document.getElementById("preview-b"),
  metaA: document.getElementById("meta-a"),
  metaB: document.getElementById("meta-b"),
  submitButton: document.getElementById("submit-button"),
  statusLine: document.getElementById("status-line"),
  compositePreview: document.getElementById("composite-preview"),
  resultEmpty: document.getElementById("result-empty"),
  resultMeta: document.getElementById("result-meta"),
  overallScore: document.getElementById("overall-score"),
  landmarkScore: document.getElementById("landmark-score"),
  warpScore: document.getElementById("warp-score"),
  textureScore: document.getElementById("texture-score"),
  agreementScore: document.getElementById("agreement-score"),
};

function setStatus(message, type = "") {
  el.statusLine.textContent = message;
  el.statusLine.className = `status-line ${type}`.trim();
}

function updateSubmitState() {
  el.submitButton.disabled = !(state.imageA && state.imageB);
}

function setImage(slot, file) {
  const isA = slot === "a";
  const preview = isA ? el.previewA : el.previewB;
  const meta = isA ? el.metaA : el.metaB;
  const dropzone = isA ? el.dropzoneA : el.dropzoneB;

  if (isA) {
    state.imageA = file;
  } else {
    state.imageB = file;
  }

  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropzone.classList.add("has-image");
  meta.textContent = file.name;
  setStatus(state.imageA && state.imageB ? "Ready to generate the composite." : "Add one more image.", "");
  updateSubmitState();
}

function wireDropzone(dropzone, input, slot) {
  input.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (file) {
      setImage(slot, file);
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files || [];
    if (file) {
      setImage(slot, file);
    }
  });
}

function setMetric(element, value) {
  element.textContent = `${Number(value).toFixed(1)}%`;
}

async function generateComposite() {
  if (!state.imageA || !state.imageB) {
    return;
  }

  el.submitButton.disabled = true;
  setStatus("Detecting faces, warping landmarks, and blending the composite...", "");

  const formData = new FormData();
  formData.append("image_a", state.imageA, state.imageA.name || "image-a.jpg");
  formData.append("image_b", state.imageB, state.imageB.name || "image-b.jpg");

  try {
    const response = await fetch("/api/v1/composite/two-face", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Could not generate composite.");
    }

    el.compositePreview.src = `${data.composite_url}?t=${Date.now()}`;
    el.compositePreview.hidden = false;
    el.resultEmpty.hidden = true;
    el.resultMeta.textContent = data.file;

    setMetric(el.overallScore, data.similarity.overall_similarity);
    setMetric(el.landmarkScore, data.similarity.landmark_geometry_similarity);
    setMetric(el.warpScore, data.similarity.warp_similarity);
    setMetric(el.textureScore, data.similarity.texture_similarity);
    setMetric(el.agreementScore, data.similarity.composite_agreement);

    setStatus("Composite generated. The score is visual similarity, not identity verification.", "success");
  } catch (error) {
    setStatus(error.message || "Could not generate composite.", "error");
  } finally {
    updateSubmitState();
  }
}

wireDropzone(el.dropzoneA, el.fileA, "a");
wireDropzone(el.dropzoneB, el.fileB, "b");
el.submitButton.addEventListener("click", generateComposite);
