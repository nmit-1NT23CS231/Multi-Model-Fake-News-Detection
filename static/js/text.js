(function () {
  "use strict";
  const { el, postJSON, setBusy, showLoading, showError, showResult } = Authen;
  const input = document.getElementById("text-input");
  const counter = document.getElementById("counter");
  const btn = document.getElementById("check-btn");
  const fileInput = document.getElementById("file-input");

  const EXAMPLES = [
    "The Federal Reserve raised its benchmark interest rate by a quarter percentage point Wednesday, continuing its fight against inflation. Fed Chair Jerome Powell stated that the committee remains committed to returning inflation to its 2 percent target. Economists widely expected the move, following months of elevated price pressures.",
    "BREAKING: Scientists confirm that 5G towers are activating dormant viruses in vaccinated people! The government is hiding this from the public. Share this before it gets taken down! The deep state does NOT want you to know that the real agenda behind 5G is population control - wake up!",
    "The World Health Organization released new guidelines on antibiotic resistance, urging governments to reduce overprescription of antimicrobials. The report, compiled by international health experts, warns that drug-resistant infections could cause 10 million deaths annually by 2050 without coordinated action.",
  ];

  const updateCounter = () => {
    const words = input.value.trim() ? input.value.trim().split(/\s+/).length : 0;
    counter.textContent = `${words.toLocaleString()} word${words === 1 ? "" : "s"}`;
  };
  input.addEventListener("input", updateCounter);

  document.querySelectorAll("[data-example]").forEach(b => b.addEventListener("click", () => {
    input.value = EXAMPLES[+b.dataset.example]; updateCounter(); input.focus();
  }));
  document.getElementById("clear-btn").addEventListener("click", () => { input.value = ""; updateCounter(); input.focus(); });
  document.getElementById("file-btn").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async () => {
    const f = fileInput.files[0]; if (!f) return;
    input.value = await f.text(); updateCounter(); fileInput.value = "";
  });

  btn.addEventListener("click", async () => {
    const text = input.value;
    if (!text.trim()) { showError("Paste some text first, then choose Check text."); input.focus(); return; }
    setBusy(btn, true); showLoading();
    try {
      const r = await postJSON("/api/text", { text });
      const fake = r.verdict === "FAKE";
      showResult({
        tone: fake ? "bad" : "ok",
        title: fake ? "Looks fake" : "Looks real",
        sub: `${r.confidence.toFixed(1)}% confident`,
        note: fake ? "The wording and style match the fake or misleading news this model learned from."
                   : "The wording and style match the real news reporting this model learned from.",
        okLabel: "Real", badLabel: "Fake", okPct: r.real_probability, badPct: r.fake_probability,
        notices: r.warning ? [{ text: r.warning }] : [],
        facts: [["Length", `${r.word_count.toLocaleString()} words`]],
        quote: "What the model read: " + r.cleaned_preview,
        caveat: "This model judges writing style and vocabulary. It cannot check whether the claims are factually true.",
      });
    } catch (e) { showError(e.message); }
    finally { setBusy(btn, false, "Check text"); }
  });
})();
