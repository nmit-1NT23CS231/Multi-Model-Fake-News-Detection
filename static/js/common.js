/* authen - shared helpers. All server data is inserted with textContent (never innerHTML),
   because email subjects/bodies come from untrusted senders. */
(function () {
  "use strict";

  function el(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  async function api(url, options) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (e) {
      throw new Error("Could not reach the authen server. Is it still running?");
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON error body */ }
    if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status}).`);
    return data;
  }

  const postJSON = (url, body) => api(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const postFile = (url, file) => {
    const fd = new FormData(); fd.append("file", file);
    return api(url, { method: "POST", body: fd });
  };

  function setBusy(btn, busy, idleLabel) {
    btn.disabled = busy;
    btn.replaceChildren(...(busy ? [el("span", { class: "spinner", "aria-hidden": "true" }), "Checking"] : [idleLabel]));
  }

  const out = () => document.getElementById("result");
  function showLoading() {
    out().replaceChildren(el("div", { class: "loading", role: "status" }, el("div", { class: "spinner" }), el("p", { text: "Checking" })));
  }
  function showError(message) {
    out().replaceChildren(el("div", { class: "notice notice--bad", role: "alert", text: message }));
  }

  /* Build the result card.
     r = { tone: 'ok'|'bad', title, sub, note, okLabel, badLabel, okPct, badPct,
           notices: [{text, bad}], facts: [[label, string|Node]], quote, caveat } */
  function showResult(r) {
    const nodes = [];
    nodes.push(el("div", { class: "verdict" },
      el("div", { class: `stamp tone-${r.tone}` },
        el("span", { class: "stamp-title", text: r.title }),
        el("span", { class: "stamp-sub", text: r.sub })),
      el("p", { class: "verdict-note", text: r.note })));

    const okBar = el("span", { class: "meter-ok", style: "width:0%" });
    const badBar = el("span", { class: "meter-bad", style: "width:0%" });
    nodes.push(el("div", { class: "meter" },
      el("div", { class: "meter-bar", role: "img", "aria-label": `${r.okLabel} ${r.okPct}%, ${r.badLabel} ${r.badPct}%` }, okBar, badBar),
      el("div", { class: "meter-legend" },
        el("span", { class: "tone-ok", text: `${r.okLabel} ${r.okPct.toFixed(1)}%` }),
        el("span", { class: "tone-bad", text: `${r.badLabel} ${r.badPct.toFixed(1)}%` }))));

    (r.notices || []).forEach(n => nodes.push(el("div", { class: "notice" + (n.bad ? " notice--bad" : ""), text: n.text })));

    if (r.facts && r.facts.length) {
      const dl = el("dl", { class: "facts" });
      r.facts.forEach(([k, v]) => dl.append(el("dt", { text: k }), el("dd", {}, v)));
      nodes.push(dl);
    }
    if (r.quote) nodes.push(el("div", { class: "quote", text: r.quote }));
    if (r.caveat) nodes.push(el("p", { class: "caveat", text: r.caveat }));

    out().replaceChildren(...nodes);
    // animate the meter after layout so the transition runs
    requestAnimationFrame(() => requestAnimationFrame(() => {
      okBar.style.width = r.okPct + "%"; badBar.style.width = r.badPct + "%";
    }));
    out().scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }

  function formatBytes(n) {
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
    return (n / 1048576).toFixed(1) + " MB";
  }

  /* drag-and-drop wiring shared by image + email pages */
  function wireDropzone(zone, input, onFile) {
    input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
    ["dragenter", "dragover"].forEach(t => zone.addEventListener(t, e => { e.preventDefault(); zone.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach(t => zone.addEventListener(t, e => { e.preventDefault(); zone.classList.remove("is-over"); }));
    zone.addEventListener("drop", e => {
      const f = e.dataTransfer.files[0];
      if (f) { try { const dt = new DataTransfer(); dt.items.add(f); input.files = dt.files; } catch (_) {} onFile(f); }
    });
  }

  window.Authen = { el, postJSON, postFile, setBusy, showLoading, showError, showResult, formatBytes, wireDropzone };
})();
