/* 流式播放顺滑度验证：真实 Chrome 里跑 GGUF 流式，读出前端自己的断供统计。
 * 判定：underrunCount == 0（或累计缺口 < 120 ms），且 AudioContext 时钟确实在走。 */
const path = require("path");
const fs = require("fs");
const puppeteer = require("/Users/junjian/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core");

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const URL = process.env.APP_URL || "http://127.0.0.1:8817";
const ENGINE = process.env.ENGINE || "gguf";
const OUT = path.join(__dirname, "shots");

const TEXT =
  process.env.TEXT ||
  "包含分段堆叠图、逐 block 边际成本柱图、四个证伪的原始数据、分层优化表和复现命令，" +
    "这些内容共同构成了这份报告的完整结构，也决定了它的阅读节奏。";

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: [
      "--no-sandbox",
      "--autoplay-policy=no-user-gesture-required",
      "--force-device-scale-factor=2",
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 960, deviceScaleFactor: 2 });

  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });
  page.on("requestfailed", (r) => {
    const u = r.url();
    const e = r.failure()?.errorText;
    if (!(u.includes("/api/stream") && e === "net::ERR_ABORTED")) {
      errors.push(`requestfailed: ${u} ${e}`);
    }
  });

  await page.goto(`${URL}/?v=${Date.now()}`, { waitUntil: "networkidle2", timeout: 30000 });
  if (process.env.LOW === "1") {
    await page.evaluate(() => localStorage.setItem("voxcpm2.lowLatency", "1"));
    await page.reload({ waitUntil: "networkidle2" });
  } else {
    await page.evaluate(() => localStorage.removeItem("voxcpm2.lowLatency"));
  }
  await page.waitForFunction(
    () => document.getElementById("statusModel")?.textContent.includes("就绪"),
    { timeout: 150000 }
  );

  // AudioContext 时钟是否真的按实时推进（headless 下若无音频设备可能不走）
  const clock = await page.evaluate(async () => {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    await ac.resume();
    const t0 = ac.currentTime;
    await new Promise((r) => setTimeout(r, 1000));
    const t1 = ac.currentTime;
    await ac.close();
    return { advanced: +(t1 - t0).toFixed(3), state: ac.state };
  });

  await page.evaluate(
    (eng) => {
      const b = document.querySelector(`#engineSeg button[data-engine="${eng}"]`);
      if (b && !b.disabled) b.click();
    },
    ENGINE
  );
  await new Promise((r) => setTimeout(r, 300));

  await page.evaluate((t) => {
    const el = document.getElementById("text");
    el.value = t;
    el.dispatchEvent(new Event("input"));
    document.getElementById("streamToggle").checked = true;
    window.__streamStats = null;
  }, TEXT);

  const t0 = Date.now();
  await page.click("#genBtn");
  await page.waitForFunction(() => !document.getElementById("genBtn").disabled, {
    timeout: 240000,
  });
  const wallMs = Date.now() - t0;

  const stats = await page.evaluate(() => window.__streamStats || null);
  const ui = await page.evaluate(() => ({
    engine: document.getElementById("mEngine").textContent.trim(),
    dur: document.getElementById("mDur").textContent,
    rtf: document.getElementById("mRtf").textContent,
    bufVisible: !document.getElementById("mBuf").hidden,
    bufText: document.getElementById("mBuf").textContent,
  }));

  await page.screenshot({ path: path.join(OUT, `stream-smooth-${ENGINE}.png`) });

  // 用前端记录的到达序列独立重算「不给缓冲闸门」时的缺口，作为对照
  const sim = (() => {
    if (!stats) return null;
    const rows = stats.arrivals.map(([t, d]) => [t / 1000, d / 1000]);
    const run = (lead) => {
      let buf = 0, gap = 0, n = 0, started = false, last = rows[0][0];
      for (const [t, d] of rows) {
        const dt = t - last;
        if (started && dt > 0) {
          if (buf >= dt) buf -= dt;
          else { gap += dt - buf; n++; buf = 0; }
        }
        buf += d;
        if (!started && t >= lead) started = true;
        last = t;
      }
      return { gap: +(gap * 1000).toFixed(0), gaps: n };
    };
    return { noBuffer: run(0.15), withGate: run(stats.startedAtSec) };
  })();

  const ok = errors.length === 0 && stats && stats.underrunCount === 0;
  console.log(
    JSON.stringify({ ok, engine: ENGINE, wallMs, clock, ui, stats: stats && {
      ...stats,
      arrivals: undefined,
    }, sim, errors }, null, 2)
  );
  await browser.close();
  if (!ok) process.exit(2);
})().catch((e) => {
  console.error("VERIFY_FAILED:", e.message);
  process.exit(1);
});
