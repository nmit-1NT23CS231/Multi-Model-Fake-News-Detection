(function () {
  "use strict";
  const { el, postFile, postJSON, setBusy, showLoading, showError, showResult, formatBytes, wireDropzone } = Authen;
  const tabs = { file: document.getElementById("tab-file"), paste: document.getElementById("tab-paste") };
  const panes = { file: document.getElementById("pane-file"), paste: document.getElementById("pane-paste") };
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  const preview = document.getElementById("preview");
  const btn = document.getElementById("check-btn");
  let mode = "file", file = null;

  function setMode(m) {
    mode = m;
    for (const k of ["file", "paste"]) {
      const on = k === m;
      tabs[k].setAttribute("aria-selected", on); tabs[k].tabIndex = on ? 0 : -1; panes[k].hidden = !on;
    }
  }
  tabs.file.addEventListener("click", () => setMode("file"));
  tabs.paste.addEventListener("click", () => setMode("paste"));
  [tabs.file, tabs.paste].forEach(t => t.addEventListener("keydown", e => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const m = mode === "file" ? "paste" : "file"; setMode(m); tabs[m].focus(); }
  }));

  function choose(f) {
    if (!/\.(eml|pdf)$/i.test(f.name)) { showError("Unsupported file type. Upload a .eml or .pdf file."); return; }
    file = f;
    document.getElementById("file-name").textContent = f.name;
    document.getElementById("file-size").textContent = formatBytes(f.size);
    zone.hidden = true; preview.classList.add("is-on");
  }
  wireDropzone(zone, input, choose);
  document.getElementById("swap-btn").addEventListener("click", () => { input.value = ""; input.click(); });

  document.getElementById("example-btn").addEventListener("click", () => {
    document.getElementById("p-from").value = "security-team@paypa1-support.com";
    document.getElementById("p-subject").value = "Urgent: your account has been suspended";
    document.getElementById("p-body").value =
      "Dear customer, we detected unusual activity and your account has been suspended. " +
      "Verify your password and banking details within 24 hours at http://secure-login.example-verify.com or your account will be permanently closed.";
  });

  const kb = n => formatBytes(n);

  btn.addEventListener("click", async () => {
    let request;
    if (mode === "file") {
      if (!file) { showError("Choose a .eml or .pdf file first."); return; }
      request = () => postFile("/api/email", file);
    } else {
      const body = document.getElementById("p-body").value;
      if (!body.trim()) { showError("Paste the message text first."); return; }
      request = () => postJSON("/api/email", {
        body, subject: document.getElementById("p-subject").value, from: document.getElementById("p-from").value });
    }
    setBusy(btn, true); showLoading();
    try {
      const r = await request();
      const fake = r.prediction === "FAKE";
      const notices = [];
      if (r.risky_attachments.length)
        notices.push({ bad: true, text: `Risky attachment${r.risky_attachments.length > 1 ? "s" : ""}: ${r.risky_attachments.join(", ")}. Do not open ${r.risky_attachments.length > 1 ? "them" : "it"}.` });

      const facts = [];
      if (r.source === "pdf") facts.push(["Document", r.subject]);
      else {
        if (r.subject) facts.push(["Subject", r.subject]);
        if (r.from) facts.push(["From", r.from]);
      }
      facts.push(["Text length", `${r.body_char_count.toLocaleString()} characters`]);
      if (r.source === "eml") {
        facts.push(["Attachments", r.attachments.length === 0 ? "None" :
          el("ul", { class: "att-list" }, r.attachments.map(a =>
            el("li", {}, a.filename, el("span", { class: "meta-row", style: "margin:0", text: kb(a.size_bytes) }),
               a.risky ? el("span", { class: "tag", text: a.macro ? "macro-enabled" : "risky type" }) : null)))]);
      }

      showResult({
        tone: fake ? "bad" : "ok",
        title: fake ? "Likely phishing" : "Looks legitimate",
        sub: `${r.confidence.toFixed(1)}% confident`,
        note: fake ? "The wording matches phishing and scam messages. Do not click links or reply."
                   : "The wording matches normal, legitimate email.",
        okLabel: "Legitimate", badLabel: "Phishing", okPct: r.legit_probability, badPct: r.fake_probability,
        notices, facts,
        quote: r.body_preview ? r.body_preview : null,
        caveat: "A model estimate, not proof. It reads the text only: it does not check sender authentication or follow links.",
      });
    } catch (e) { showError(e.message); }
    finally { setBusy(btn, false, "Check email"); }
  });
})();
