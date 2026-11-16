(function () {
  "use strict";
  const { el, postFile, setBusy, showLoading, showError, showResult, formatBytes, wireDropzone } = Authen;
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  const preview = document.getElementById("preview");
  const img = document.getElementById("preview-img");
  const nameEl = document.getElementById("file-name");
  const btn = document.getElementById("check-btn");
  let file = null, objectUrl = null;

  function choose(f) {
    if (!f.type.startsWith("image/")) { showError("That file is not an image. Choose a JPG, PNG, BMP or WebP file."); return; }
    file = f;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);
    img.src = objectUrl;
    nameEl.textContent = `${f.name} (${formatBytes(f.size)})`;
    zone.hidden = true; preview.classList.add("is-on");
    btn.disabled = !window.AUTHEN_MODEL_READY;
  }
  wireDropzone(zone, input, choose);
  document.getElementById("swap-btn").addEventListener("click", () => { input.value = ""; input.click(); });

  btn.addEventListener("click", async () => {
    if (!file) return;
    setBusy(btn, true); showLoading();
    try {
      const r = await postFile("/api/image", file);
      const fake = r.classification === "FAKE";
      const p = r.ai_probability_percent;
      const low = r.confidence === "Low";
      showResult({
        tone: fake ? "bad" : "ok",
        title: fake ? "Likely AI-generated" : "Likely authentic",
        sub: `${r.confidence} confidence`,
        note: fake ? "This image has traits the model associates with AI-generated or faked pictures."
                   : "This image has traits the model associates with real photographs.",
        okLabel: "Authentic", badLabel: "AI-generated", okPct: 100 - p, badPct: p,
        notices: low ? [{ text: "The model is close to undecided on this one. Treat the result as a weak hint." }] : [],
        facts: [
          ["Size", `${r.width} \u00d7 ${r.height} px (${r.format})`],
          ["Model saw", el("div", { class: "model-view" },
            el("img", { src: r.model_view, alt: `The ${r.model_input_size} by ${r.model_input_size} pixel version the model analysed` }),
            el("small", { text: `Your image shrunk to ${r.model_input_size}\u00d7${r.model_input_size} pixels. Fine detail is lost at this size.` }))],
        ],
        caveat: "A model estimate, not proof. Edited, compressed or screenshotted images can fool it either way.",
      });
    } catch (e) { showError(e.message); }
    finally { setBusy(btn, false, "Check image"); }
  });
})();
