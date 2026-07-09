const apiBase = "/api/v1";

const pages = {
  url: document.getElementById("page-url"),
  image: document.getElementById("page-image"),
  video: document.getElementById("page-video"),
  text: document.getElementById("page-text"),
};

const menuItems = Array.from(document.querySelectorAll(".menu-item"));
const loadingOverlay = document.getElementById("loadingOverlay");
const toastRoot = document.getElementById("toastRoot");
const themeToggle = document.getElementById("themeToggle");

const urlInput = document.getElementById("urlInput");
const imageInput = document.getElementById("imageInput");
const videoInput = document.getElementById("videoInput");
const textInput = document.getElementById("textInput");
const imagePreviewWrap = document.getElementById("imagePreviewWrap");
const imagePreview = document.getElementById("imagePreview");
const imageOverlayWrap = document.getElementById("imageOverlayWrap");
const imageOverlay = document.getElementById("imageOverlay");
const videoPreviewWrap = document.getElementById("videoPreviewWrap");
const videoPreview = document.getElementById("videoPreview");
const videoDropzone = document.getElementById("videoDropzone");
const dropzone = document.getElementById("dropzone");

const urlResult = document.getElementById("urlResult");
const imageResult = document.getElementById("imageResult");
const videoResult = document.getElementById("videoResult");
const videoTimeline = document.getElementById("videoTimeline");
const textResult = document.getElementById("textResult");

function switchPage(page) {
  menuItems.forEach((item) => {
    const isActive = item.dataset.page === page;
    item.classList.toggle("active", isActive);
  });
  Object.keys(pages).forEach((key) => {
    pages[key].classList.toggle("active", key === page);
  });
}

menuItems.forEach((item) => {
  item.addEventListener("click", () => switchPage(item.dataset.page));
});

function showLoading(show) {
  loadingOverlay.classList.toggle("hidden", !show);
}

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastRoot.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 2600);
}

function verdictBadge(label) {
  const normalized = String(label || "").toLowerCase();
  const cls = normalized.includes("original") || normalized === "real" ? "real" : normalized.includes("ai") || normalized === "fake" ? "fake" : "verify";
  return `<span class="badge ${cls}">${label}</span>`;
}

function moduleConfidence(moduleScore) {
  return `${(Math.abs(Number(moduleScore) - 0.5) * 200).toFixed(1)}%`;
}

function listReasons(reasons) {
  if (!reasons || reasons.length === 0) {
    return "<li>No major reasons detected.</li>";
  }
  return reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("");
}

function renderEvidence(evidenceRows) {
  if (!evidenceRows || evidenceRows.length === 0) {
    return "<li>No internet evidence snippets available.</li>";
  }

  const topRows = evidenceRows
    .filter((row) => row && row.snippet)
    .slice(0, 4)
    .map((row) => {
      const stance = String(row.stance || "neutral").toUpperCase();
      const source = String(row.source || "web");
      const relevance = Number(row.relevance || 0).toFixed(2);
      return `<li><strong>${escapeHtml(stance)}</strong> | ${escapeHtml(source)} | relevance ${relevance}<br>${escapeHtml(String(row.snippet || ""))}</li>`;
    });

  if (topRows.length === 0) {
    return "<li>Internet checks were inconclusive for this claim.</li>";
  }

  return topRows.join("");
}

function escapeHtml(input) {
  return String(input)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function highlightTerms(text, terms) {
  const safeText = escapeHtml(text || "");
  if (!terms || terms.length === 0) return safeText;

  let highlighted = safeText;
  terms.forEach((term) => {
    if (!term) return;
    const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(`(${escapedTerm})`, "gi");
    highlighted = highlighted.replace(pattern, "<mark>$1</mark>");
  });

  return highlighted;
}

async function runPrediction(payload) {
  showLoading(true);
  try {
    const response = await fetch(`${apiBase}/predict`, {
      method: "POST",
      body: payload,
    });

    let data;
    if (response.headers.get("content-type")?.includes("application/json")) {
      data = await response.json();
    } else {
      // Try parsing JSON, but gracefully fallback to text for HTML/error pages
      try {
        data = await response.json();
      } catch (e) {
        const text = await response.text();
        if (!response.ok) throw new Error(text || "Prediction failed with non-JSON response");
        // create a minimal object when server returned plain text on success
        data = { detail: text };
      }
    }

    if (!response.ok) {
      // Prefer structured error detail, otherwise surface raw text
      throw new Error((data && data.detail) || "Prediction failed");
    }

    showToast("Analysis completed", "success");
    return data;
  } catch (error) {
    showToast(error.message, "error");
    throw error;
  } finally {
    showLoading(false);
  }
}

async function runImageAuthenticity(payload) {
  showLoading(true);
  try {
    const response = await fetch(`${apiBase}/image-authenticity`, {
      method: "POST",
      body: payload,
    });

    let data;
    try {
      data = await response.json();
    } catch (e) {
      const text = await response.text();
      if (!response.ok) throw new Error(text || "Image authenticity analysis failed");
      data = { detail: text };
    }
    if (!response.ok) {
      throw new Error((data && data.detail) || "Image authenticity analysis failed");
    }

    showToast("Image authenticity analysis completed", "success");
    return data;
  } catch (error) {
    showToast(error.message, "error");
    throw error;
  } finally {
    showLoading(false);
  }
}

async function runVideoAuthenticity(payload) {
  showLoading(true);
  try {
    const response = await fetch(`${apiBase}/video-authenticity`, {
      method: "POST",
      body: payload,
    });

    let data;
    try {
      data = await response.json();
    } catch (e) {
      const text = await response.text();
      if (!response.ok) throw new Error(text || "Video authenticity analysis failed");
      data = { detail: text };
    }
    if (!response.ok) {
      throw new Error((data && data.detail) || "Video authenticity analysis failed");
    }

    showToast("Video authenticity analysis completed", "success");
    return data;
  } catch (error) {
    showToast(error.message, "error");
    throw error;
  } finally {
    showLoading(false);
  }
}

function renderUrlResult(result) {
  const moduleScore = Number(result?.module_scores?.url ?? 0.5);
  const moduleReasons = result?.module_reasons?.url || [];
  const factReasons = result?.module_reasons?.fact_check || [];
  const overallReasons = result?.reasons || [];
  const details = result?.module_details?.url || {};
  const factDetails = result?.module_details?.fact_check || {};
  const factEvidence = factDetails.evidence || [];
  const summary = result?.summary || "Summary unavailable.";
  const verdictReason = result?.verdict_reason || "No final verdict reason available.";

  urlResult.innerHTML = `
    <div class="result-header">
      ${verdictBadge(result.label)}
      <div class="conf">Confidence: ${(result.confidence * 100).toFixed(2)}%</div>
    </div>
    <p class="meta"><strong>Summary:</strong> ${escapeHtml(summary)}</p>
    <p class="meta"><strong>Verdict reason:</strong> ${escapeHtml(verdictReason)}</p>
    <p class="meta">URL module confidence: ${moduleConfidence(moduleScore)}</p>
    <p class="meta">Domain: ${escapeHtml(details.domain || "Unknown")}</p>
    <p class="meta"><strong>Overall reasoning</strong></p>
    <ul class="reason-list">${listReasons(overallReasons)}</ul>
    <p class="meta"><strong>Internet fact-check</strong> (claims checked: ${escapeHtml(factDetails.claims_checked ?? 0)}, overlap: ${escapeHtml(factDetails.avg_overlap ?? 0)})</p>
    <ul class="reason-list">${listReasons(factReasons)}</ul>
    <p class="meta"><strong>Evidence snippets</strong></p>
    <ul class="reason-list evidence-list">${renderEvidence(factEvidence)}</ul>
    <p class="meta"><strong>URL-specific cues</strong></p>
    <ul class="reason-list">${listReasons(moduleReasons)}</ul>
  `;
  urlResult.classList.remove("hidden");
}

function renderImageResult(result) {
  const label = result?.label || "Original";
  const confidence = Number(result?.confidence ?? 0);
  const probabilities = result?.probabilities || {};
  const reasons = result?.reasons || [];
  const suspiciousRegions = result?.suspicious_regions || [];
  const overlayImageSrc = result?.overlay_image || "";

  imageResult.classList.remove("auth-ai", "auth-original");
  imageResult.classList.add(label === "AI Generated" ? "auth-ai" : "auth-original");

  imageResult.innerHTML = `
    <div class="result-header">
      ${verdictBadge(label)}
      <div class="conf">Confidence: ${(confidence * 100).toFixed(2)}%</div>
    </div>
    <p class="meta"><strong>AI Generated:</strong> ${(Number(probabilities.AI_GENERATED || 0) * 100).toFixed(1)}%</p>
    <p class="meta"><strong>Original:</strong> ${(Number(probabilities.ORIGINAL || 0) * 100).toFixed(1)}%</p>
    <ul class="reason-list">${listReasons(reasons)}</ul>
    <p class="meta"><strong>Suspicious regions</strong></p>
    <ul class="reason-list region-list">${renderRegions(suspiciousRegions)}</ul>
  `;
  imageResult.classList.remove("hidden");

  if (overlayImageSrc) {
    imageOverlay.src = overlayImageSrc;
    imageOverlayWrap.classList.remove("hidden");
  } else {
    imageOverlay.src = "";
    imageOverlayWrap.classList.add("hidden");
  }
}

function renderVideoResult(result) {
  const label = result?.label || "Real Video";
  const confidence = Number(result?.confidence ?? 0);
  const frameScores = Array.isArray(result?.frame_scores) ? result.frame_scores : [];
  const suspiciousFrames = new Set(result?.suspicious_frames || []);

  videoResult.classList.remove("auth-ai", "auth-original");
  videoResult.classList.add(label.toLowerCase().includes("fake") ? "auth-ai" : "auth-original");

  videoResult.innerHTML = `
    <div class="result-header">
      ${verdictBadge(label)}
      <div class="conf">Confidence: ${(confidence * 100).toFixed(2)}%</div>
    </div>
    <p class="meta"><strong>Suspicious frames:</strong> ${suspiciousFrames.size ? Array.from(suspiciousFrames).join(", ") : "None"}</p>
    <p class="meta"><strong>Frame scores:</strong> ${frameScores.length}</p>
  `;
  videoResult.classList.remove("hidden");

  videoTimeline.innerHTML = "";
  if (frameScores.length) {
    frameScores.forEach((score, index) => {
      const mark = document.createElement("div");
      mark.className = `timeline-mark ${suspiciousFrames.has(index) ? "suspicious" : ""}`;
      mark.title = `Frame ${index + 1}: ${Number(score).toFixed(3)}`;
      videoTimeline.appendChild(mark);
    });
    videoTimeline.classList.remove("hidden");
  } else {
    videoTimeline.classList.add("hidden");
  }
}

function renderRegions(regions) {
  if (!regions || regions.length === 0) {
    return "<li>No suspicious regions highlighted.</li>";
  }

  return regions
    .map((region, index) => {
      const score = Number(region?.score || 0);
      return `<li>Region ${index + 1}: ${escapeHtml(region.x)}, ${escapeHtml(region.y)}, ${escapeHtml(region.width)}x${escapeHtml(region.height)} - score ${(score * 100).toFixed(1)}%</li>`;
    })
    .join("");
}

function renderTextResult(result, sourceText) {
  const moduleScore = Number(result?.module_scores?.text ?? 0.5);
  const moduleReasons = result?.module_reasons?.text || [];
  const factReasons = result?.module_reasons?.fact_check || [];
  const details = result?.module_details?.text || {};
  const factDetails = result?.module_details?.fact_check || {};
  const factEvidence = factDetails.evidence || [];
  const suspiciousTerms = details.suspicious_terms || [];

  textResult.innerHTML = `
    <div class="result-header">
      ${verdictBadge(result.label)}
      <div class="conf">Confidence: ${(result.confidence * 100).toFixed(2)}%</div>
    </div>
    <p class="meta">Text module confidence: ${moduleConfidence(moduleScore)}</p>
    <ul class="reason-list">${listReasons(moduleReasons)}</ul>
    <p class="meta"><strong>Internet fact-check</strong> (claims checked: ${escapeHtml(factDetails.claims_checked ?? 0)}, overlap: ${escapeHtml(factDetails.avg_overlap ?? 0)})</p>
    <ul class="reason-list">${listReasons(factReasons)}</ul>
    <p class="meta"><strong>Evidence snippets</strong></p>
    <ul class="reason-list evidence-list">${renderEvidence(factEvidence)}</ul>
    <div class="highlight-wrap">
      <strong>Suspicious word highlight</strong>
      <p>${highlightTerms(sourceText, suspiciousTerms)}</p>
    </div>
  `;
  textResult.classList.remove("hidden");
}

document.getElementById("analyzeUrl").addEventListener("click", async () => {
  if (!urlInput.value.trim()) {
    showToast("Please enter a URL first", "error");
    return;
  }

  const formData = new FormData();
  formData.append("url", urlInput.value.trim());
  const result = await runPrediction(formData);
  renderUrlResult(result);
});

document.getElementById("clearUrl").addEventListener("click", () => {
  urlInput.value = "";
  urlResult.classList.add("hidden");
  showToast("URL form reset", "success");
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = event.dataTransfer.files?.[0];
  if (!file) return;
  imageInput.files = event.dataTransfer.files;
  previewImage(file);
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (file) previewImage(file);
});

videoInput.addEventListener("change", () => {
  const file = videoInput.files?.[0];
  if (file) previewVideo(file);
});

function previewImage(file) {
  const reader = new FileReader();
  reader.onload = () => {
    imagePreview.src = String(reader.result || "");
    imagePreviewWrap.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

function previewVideo(file) {
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
  videoPreviewWrap.classList.remove("hidden");
}

document.getElementById("analyzeImage").addEventListener("click", async () => {
  const file = imageInput.files?.[0];
  if (!file) {
    showToast("Please upload an image first", "error");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);
  const result = await runImageAuthenticity(formData);
  renderImageResult(result);
});

document.getElementById("clearImage").addEventListener("click", () => {
  imageInput.value = "";
  imagePreview.src = "";
  imagePreviewWrap.classList.add("hidden");
  imageOverlay.src = "";
  imageOverlayWrap.classList.add("hidden");
  imageResult.classList.add("hidden");
  imageResult.classList.remove("auth-ai", "auth-original");
  showToast("Image form reset", "success");
});

document.getElementById("analyzeVideo").addEventListener("click", async () => {
  const file = videoInput.files?.[0];
  if (!file) {
    showToast("Please upload a video first", "error");
    return;
  }

  const formData = new FormData();
  formData.append("video", file);
  const result = await runVideoAuthenticity(formData);
  renderVideoResult(result);
});

document.getElementById("clearVideo").addEventListener("click", () => {
  videoInput.value = "";
  videoPreview.src = "";
  videoPreviewWrap.classList.add("hidden");
  videoResult.classList.add("hidden");
  videoTimeline.classList.add("hidden");
  videoTimeline.innerHTML = "";
  showToast("Video form reset", "success");
});

document.getElementById("analyzeText").addEventListener("click", async () => {
  const value = textInput.value.trim();
  if (!value) {
    showToast("Please enter text first", "error");
    return;
  }

  const formData = new FormData();
  formData.append("text", value);
  const result = await runPrediction(formData);
  renderTextResult(result, value);
});

document.getElementById("clearText").addEventListener("click", () => {
  textInput.value = "";
  textResult.classList.add("hidden");
  showToast("Text form reset", "success");
});

themeToggle.addEventListener("click", () => {
  const isDark = document.body.classList.toggle("dark");
  themeToggle.textContent = isDark ? "Dark Mode" : "Light Mode";
});

switchPage("url");
