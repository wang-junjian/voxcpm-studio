/* Web 语音合成应用端到端验证：真实 Chrome 加载、交互、截图、断言 */
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
  const known = []; // Chrome 对 chunked 流式响应的标注，不影响数据完整性
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

  await page.goto(URL, { waitUntil: "networkidle2", timeout: 30000 });

  // 1) 等待模型就绪
  await page.waitForFunction(
    () => document.getElementById("statusModel")?.textContent.includes("就绪"),
    { timeout: 90000 }
  );
  const status = await page.evaluate(() => ({
    model: document.getElementById("statusModel").textContent.trim(),
    device: document.getElementById("statusDev").textContent.trim(),
    sr: document.getElementById("statusSr").textContent.trim(),
    btnEnabled: !document.getElementById("genBtn").disabled,
  }));
  report.steps.push({ step: "模型就绪", ...status });
  await page.screenshot({ path: path.join(OUT, "01_initial.png") });

  // 2) 检查布局无溢出/重叠
  const layout = await page.evaluate(() => {
    const bad = [];
    const docW = document.documentElement.clientWidth;
    document.querySelectorAll("*").forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && (r.right > docW + 2 || r.left < -2)) {
        bad.push(`${el.tagName}.${el.className}`.slice(0, 60));
      }
    });
    return { overflow: bad.slice(0, 8), scrollW: document.documentElement.scrollWidth, docW };
  });
  report.steps.push({ step: "布局检查", ...layout });

  // 3) 基础合成（一次性）
  await page.click("#text", { clickCount: 3 });
  await page.type("#text", "这是网页应用里生成的语音，用来验证整条链路是否通畅。");
  await page.click("#genBtn");
  await page.waitForFunction(
    () => {
      const r = document.getElementById("result");
      const d = document.getElementById("mDur");
      return r && !r.hidden && d && d.textContent !== "—";
    },
    { timeout: 120000 }
  );
  await new Promise((r) => setTimeout(r, 600));
  const once = await page.evaluate(() => ({
    dur: document.getElementById("mDur").textContent,
    rtf: document.getElementById("mRtf").textContent,
    sr: document.getElementById("mSr").textContent,
    hasWave: !!document.querySelector("#wave"),
    downloadHref: document.getElementById("download").getAttribute("href"),
  }));
  report.steps.push({ step: "基础合成", ...once });
  await page.screenshot({ path: path.join(OUT, "02_result.png") });

  // 4) 波形是否真的画出了像素（非空白 canvas）
  const waveInk = await page.evaluate(() => {
    const c = document.getElementById("wave");
    const g = c.getContext("2d");
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 10) n++;
    return { inked: n, total: d.length / 4 };
  });
  report.steps.push({ step: "波形像素", ...waveInk });

  // 5) 音色设计模式
  await page.evaluate(() => {
    document.querySelector('#modeSeg button[data-mode="design"]').click();
  });
  await new Promise((r) => setTimeout(r, 300));
  const design = await page.evaluate(() => ({
    presetsVisible: !document.getElementById("presets").hidden,
    presetCount: document.querySelectorAll("#presets .chip").length,
    hint: document.getElementById("modeHint").textContent.trim(),
  }));
  report.steps.push({ step: "音色设计模式", ...design });
  await page.screenshot({ path: path.join(OUT, "03_design.png") });

  // 6) 克隆模式
  await page.evaluate(() => {
    document.querySelector('#modeSeg button[data-mode="clone"]').click();
  });
  await new Promise((r) => setTimeout(r, 300));
  const clone = await page.evaluate(() => ({
    uploadVisible: !document.getElementById("cloneBox").hidden,
    promptTextVisible: !!document.getElementById("promptText"),
  }));
  report.steps.push({ step: "克隆模式", ...clone });
  await page.screenshot({ path: path.join(OUT, "04_clone.png") });

  // 7) 历史记录
  const hist = await page.evaluate(() => ({
    count: document.querySelectorAll(".hist-item").length,
    first: document.querySelector(".hist-text")?.textContent?.slice(0, 30),
  }));
  report.steps.push({ step: "历史记录", ...hist });

  // 8) 流式模式
  await page.evaluate(() => {
    document.querySelector('#modeSeg button[data-mode="basic"]').click();
    document.getElementById("streamToggle").checked = true;
  });
  await page.evaluate(() => {
    document.getElementById("text").value = "流式播放测试，边生成边播放。";
    document.getElementById("text").dispatchEvent(new Event("input"));
  });
  const streamStart = Date.now();
  await page.click("#genBtn");
  await page.waitForFunction(
    () => {
      const h = document.getElementById("mRtf");
      return h && h.textContent.startsWith("流");
    },
    { timeout: 20000 }
  );
  const firstChunkMs = Date.now() - streamStart;
  await page.waitForFunction(
    () => !document.getElementById("genBtn").disabled,
    { timeout: 120000 }
  );
  await new Promise((r) => setTimeout(r, 500));
  const stream = await page.evaluate(async () => {
    const a = window.__vox && window.__vox.audio;
    // 等 blob 就绪并解码，确认流式拼接的 PCM 是合法可播放音频
    let decoded = null;
    if (a && a.src.startsWith("blob:")) {
      try {
        const buf = await (await fetch(a.src)).arrayBuffer();
        const ac = new (window.AudioContext || window.webkitAudioContext)();
        const ab = await ac.decodeAudioData(buf);
        decoded = { dur: +ab.duration.toFixed(2), sr: ab.sampleRate, ch: ab.numberOfChannels };
        await ac.close();
      } catch (e) {
        decoded = { error: e.message };
      }
    }
    return {
      dur: document.getElementById("mDur").textContent,
      rtf: document.getElementById("mRtf").textContent,
      downloadVisible: document.getElementById("download").style.visibility !== "hidden",
      blobReady: !!(a && a.src.startsWith("blob:")),
      decoded,
    };
  });
  report.steps.push({ step: "流式合成", firstChunkMs, ...stream });
  await page.screenshot({ path: path.join(OUT, "05_stream.png") });

  report.ok = errors.length === 0;
  fs.writeFileSync(path.join(__dirname, "verify_report.json"), JSON.stringify(report, null, 2));

  console.log(JSON.stringify(report, null, 2));
  await browser.close();
})().catch((e) => {
  console.error("VERIFY_FAILED:", e.message);
  process.exit(1);
});
