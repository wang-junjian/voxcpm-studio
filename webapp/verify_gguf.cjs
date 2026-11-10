/* GGUF 引擎接入验证：真实 Chrome 里切到 GGUF、跑一次性与流式、查尾音是否完整 */
const path = require("path");
const fs = require("fs");

const PPTR = "/Users/junjian/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const URL = process.env.APP_URL || "http://127.0.0.1:8817";
const OUT = path.join(__dirname, "shots");

const puppeteer = require(PPTR);

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: [
      "--no-sandbox",
      "--autoplay-policy=no-user-gesture-required",
      "--mute-audio",
      "--force-device-scale-factor=2",
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 960, deviceScaleFactor: 2 });

  const errors = [];
  const known = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });
  page.on("requestfailed", (r) => {
    const u = r.url();
    const e = r.failure()?.errorText;
    if (u.includes("/api/stream") && e === "net::ERR_ABORTED") {
      known.push(`stream chunked 结束标记 (${e})`);
    } else {
      errors.push(`requestfailed: ${u} ${e}`);
    }
  });

  const report = { url: URL, steps: [], errors, known };
  const step = (name, data) => report.steps.push({ step: name, ...data });

  await page.goto(URL, { waitUntil: "networkidle2", timeout: 30000 });
  await page.waitForFunction(
    () => document.getElementById("statusModel")?.textContent.includes("就绪"),
    { timeout: 120000 }
  );

  // 1) 三个引擎按钮都在，GGUF 可用
  const eng = await page.evaluate(() => {
    const seg = document.getElementById("engineSeg");
    return {
      buttons: [...seg.querySelectorAll("button")].map((b) => ({
        engine: b.dataset.engine,
        pressed: b.getAttribute("aria-pressed"),
        disabled: b.disabled,
      })),
    };
  });
  step("引擎控件", eng);

  await page.evaluate(() => {
    const b = document.querySelector('#engineSeg button[data-engine="gguf"]');
    if (b && !b.disabled) b.click();
  });
  await new Promise((r) => setTimeout(r, 300));
  const afterSwitch = await page.evaluate(() => ({
    pressed: document.querySelector('#engineSeg button[data-engine="gguf"]').getAttribute("aria-pressed"),
    hint: document.getElementById("engineHint").textContent.trim(),
    spec: document.getElementById("specEngine").textContent.trim(),
    status: document.getElementById("statusDev").textContent.trim(),
  }));
  step("切换到 GGUF", afterSwitch);
  await page.screenshot({ path: path.join(OUT, "gguf_01_engine.png") });

  // 2) 一次性合成
  await page.evaluate(() => {
    const t = document.getElementById("text");
    t.value = "这段语音由 GGUF 八位量化引擎在本地合成，底层是 llama.cpp 的 Metal 后端。";
    t.dispatchEvent(new Event("input"));
    document.getElementById("streamToggle").checked = false;
  });
  const t1 = Date.now();
  await page.click("#genBtn");
  await page.waitForFunction(
    () => {
      const r = document.getElementById("result");
      const d = document.getElementById("mDur");
      return r && !r.hidden && d && d.textContent !== "—";
    },
    { timeout: 180000 }
  );
  await new Promise((r) => setTimeout(r, 500));
  const once = await page.evaluate(() => ({
    engine: document.getElementById("mEngine").textContent.trim(),
    dur: document.getElementById("mDur").textContent,
    rtf: document.getElementById("mRtf").textContent,
    sr: document.getElementById("mSr").textContent,
    downloadHref: document.getElementById("download").getAttribute("href"),
  }));
  step("GGUF 一次性合成", { ...once, wallMs: Date.now() - t1 });
  await page.screenshot({ path: path.join(OUT, "gguf_02_once.png") });

  // 3) 流式：首包 + 尾音
  await page.evaluate(() => {
    const t = document.getElementById("text");
    t.value = "流式播放的问题是尾部的几个字始终没有读，这句话的最后一个字必须完整念出来。";
    t.dispatchEvent(new Event("input"));
    document.getElementById("streamToggle").checked = true;
  });
  const t2 = Date.now();
  await page.click("#genBtn");
  await page.waitForFunction(
    () => {
      const m = window.__vox?.state?.streamMarks;
      return m && m.firstChunkMs != null;
    },
    { timeout: 60000 }
  );
  const firstChunkMs = Date.now() - t2;
  await page.waitForFunction(
    () => !document.getElementById("genBtn").disabled,
    { timeout: 180000 }
  );
  await new Promise((r) => setTimeout(r, 3000));

  const stream = await page.evaluate(async () => {
    const a = window.__vox && window.__vox.audio;
    const out = {
      engine: document.getElementById("mEngine").textContent.trim(),
      dur: document.getElementById("mDur").textContent,
      rtf: document.getElementById("mRtf").textContent,
      marks: window.__vox?.state?.streamMarks || null,
      blobReady: !!(a && a.src.startsWith("blob:")),
    };
    if (!out.blobReady) return out;
    const buf = await (await fetch(a.src)).arrayBuffer();
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const ab = await ac.decodeAudioData(buf);
    const d = ab.getChannelData(0);
    const sr = ab.sampleRate;
    const tailN = Math.floor(0.3 * sr);
    const tail = d.subarray(d.length - tailN);
    let sum = 0;
    for (let i = 0; i < tail.length; i++) sum += tail[i] * tail[i];
    out.decoded = { dur: +ab.duration.toFixed(2), sr, ch: ab.numberOfChannels };
    out.tailRms = +Math.sqrt(sum / tail.length).toFixed(5);
    let best = 0;
    const w = Math.floor(0.02 * sr);
    for (let s = d.length - Math.floor(0.9 * sr); s + w < d.length; s += w) {
      let acc = 0;
      for (let i = s; i < s + w; i++) acc += d[i] * d[i];
      best = Math.max(best, Math.sqrt(acc / w));
    }
    out.tailPeakRms = +best.toFixed(4);
    await ac.close();
    return out;
  });
  step("GGUF 流式合成", { firstChunkMs, ...stream });
  await page.screenshot({ path: path.join(OUT, "gguf_03_stream.png") });

  // 4) 历史记录引擎标记
  const hist = await page.evaluate(() => ({
    count: document.querySelectorAll(".hist-item").length,
    engs: [...document.querySelectorAll(".hist-eng")].slice(0, 6).map((e) => e.textContent),
  }));
  step("历史记录", hist);

  report.ok = errors.length === 0;
  fs.writeFileSync(path.join(__dirname, "verify_gguf_report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  await browser.close();
})().catch((e) => {
  console.error("VERIFY_FAILED:", e.message);
  process.exit(1);
});
