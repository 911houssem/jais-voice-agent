/* ===== Jais Voice — app.js ===== */
const API = "/api";

// ---- Status pill / mode ----
async function loadStatus() {
  try {
    const r = await fetch(`${API}/status`);
    const d = await r.json();
    const pill = document.getElementById("status-pill");
    const modeLabel = document.getElementById("mode-label");
    if (d.mode === "demo") {
      pill.textContent = "وضع Demo";
      pill.className = "pill pill-demo";
    } else {
      pill.textContent = "وضع Real";
      pill.className = "pill pill-ok";
    }
    if (modeLabel) modeLabel.textContent = d.mode;
    const el = document.getElementById("status-json");
    if (el) el.textContent = JSON.stringify(d, null, 2);
  } catch (e) {
    const el = document.getElementById("status-json");
    if (el) el.textContent = "تعذّر الاتصال بالخادم: " + e.message;
  }
}
loadStatus();

// ---- Chat demo ----
const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const demoWave = document.getElementById("demo-wave");

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + (role === "agent" ? "agent" : "user");
  div.innerHTML = `<div class="bubble"></div><div class="meta">${role === "agent" ? "وكيل" : "أنت"} · just now</div>`;
  chatLog.appendChild(div);
  const bubble = div.querySelector(".bubble");
  // typing effect
  let i = 0;
  const step = () => {
    if (i <= text.length) {
      bubble.textContent = text.slice(0, i);
      i++;
      setTimeout(step, role === "agent" ? 18 : 8);
    }
  };
  step();
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

function setTimings(t) {
  document.getElementById("t-asr").textContent = (t.asr_ms ?? 0) + " ms";
  document.getElementById("t-llm").textContent = (t.llm_ms ?? 0) + " ms";
  document.getElementById("t-tts").textContent = (t.tts_ms ?? 0) + " ms";
  document.getElementById("t-total").textContent = (t.total_ms ?? 0) + " ms";
  document.getElementById("demo-latency").textContent = (t.total_ms ?? 0) + " ms";
}

async function sendText(text) {
  if (!text.trim()) return;
  addMsg("user", text);
  chatInput.value = "";
  demoWave.classList.add("active");
  try {
    const r = await fetch(`${API}/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const d = await r.json();
    addMsg("agent", d.reply_text);
    setTimings(d.timings);
    if (d.audio_b64) {
      const audio = document.getElementById("audio-player");
      audio.src = `data:${d.mime};base64,${d.audio_b64}`;
      audio.play().catch(() => {});
      document.getElementById("audio-note").textContent = `مدة الصوت: ${d.duration_s}s · ${d.demo ? "demo tone" : "صوت عربي حقيقي"}`;
    }
  } catch (e) {
    addMsg("agent", "تعذّر الاتصال بالخادم: " + e.message);
  } finally {
    setTimeout(() => demoWave.classList.remove("active"), 600);
  }
}

sendBtn.addEventListener("click", () => sendText(chatInput.value));
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendText(chatInput.value);
});

// mic button: real microphone recording (tap to talk, tap again to stop)
const prompts = [
  "أريد حجز موعد",
  "ما هي ساعات العمل؟",
  "أريد إلغاء موعدي",
  "كم تكلفة الاستشارة؟",
  "أريد التحدث مع موظف",
];
let recorder = null, recChunks = [], recording = false;

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = [];
    const opts = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? { mimeType: "audio/webm;codecs=opus" } : {};
    recorder = new MediaRecorder(stream, opts);
    recorder.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      await handleVoice(new Blob(recChunks, { type: recorder.mimeType || "audio/webm" }));
    };
    recorder.start();
    recording = true;
    document.getElementById("audio-note").textContent = "🔴 أسمعك... اضغط على الميكروفون مرة أخرى للإيقاف والإرسال";
  } catch (e) {
    // microphone unavailable -> fall back to a demo prompt
    const p = prompts[Math.floor(Math.random() * prompts.length)];
    chatInput.value = p;
    sendText(p);
  }
}

async function handleVoice(blob) {
  demoWave.classList.add("active");
  const holder = addMsg("user", "🎙️ (رسالة صوتية)");
  try {
    const fd = new FormData();
    fd.append("file", blob, "speech.webm");
    const r = await fetch(`${API}/asr`, { method: "POST", body: fd });
    const a = await r.json();
    const text = (a.text || "").trim();
    if (!text) {
      holder.querySelector(".bubble").textContent = "🎙️ (لم أسمع كلامًا واضحًا)";
      addMsg("agent", "ما سمعت كلامًا واضحًا، جرب مرة ثانية.");
      return;
    }
    holder.querySelector(".bubble").textContent = text; // real transcript
    await sendText(text);
  } catch (e) {
    addMsg("agent", "تعذّر التعرّف على الصوت: " + e.message);
  } finally {
    setTimeout(() => demoWave.classList.remove("active"), 600);
    document.getElementById("audio-note").textContent = "";
  }
}

document.getElementById("mic-btn").addEventListener("click", () => {
  if (recording && recorder && recorder.state === "recording") {
    recorder.stop();
    recording = false;
  } else {
    startRecording();
  }
});

// ---- Dashboard demo data ----
const days = ["أحد", "إثنين", "ثلاثاء", "أربعاء", "خميس", "جمعة", "سبت"];
const vals = [42, 68, 55, 80, 72, 30, 48];
const maxV = Math.max(...vals);
const bc = document.getElementById("bar-chart");
bc.innerHTML = vals.map((v, i) => `
  <div class="bar-col">
    <div class="bar" style="height:${(v / maxV) * 100}%"></div>
    <div class="bar-label">${days[i]}</div>
  </div>
`).join("");

const calls = [
  ["أحمد السعدي", "4:12", "حجز موعد", "920ms"],
  ["فاطمة علي", "2:48", "استفسار", "840ms"],
  ["خالد المطيري", "6:05", "إلغاء", "1.1s"],
  ["نورة الحربي", "1:30", "تأهيل", "760ms"],
  ["محمد العتيبي", "5:22", "حجز موعد", "990ms"],
];
const tbody = document.getElementById("calls-tbody");
tbody.innerHTML = calls.map(c => `
  <tr><td>${c[0]}</td><td>${c[1]}</td><td>${c[2]}</td><td>${c[3]}</td></tr>
`).join("");

// ---- Hero timer ----
let t = 42;
setInterval(() => {
  t++;
  const m = String(Math.floor(t / 60)).padStart(2, "0");
  const s = String(t % 60).padStart(2, "0");
  const el = document.getElementById("hero-timer");
  if (el) el.textContent = `${m}:${s}`;
}, 1000);

// ---- KPI live-ish jitter ----
setInterval(() => {
  const a = document.getElementById("kpi-active");
  if (a) a.textContent = 10 + Math.floor(Math.random() * 8);
  const m = document.getElementById("kpi-mins");
  if (m) m.textContent = (1800 + Math.floor(Math.random() * 120)).toLocaleString();
}, 4000);
