/* 缓冲阶段的界面截图 + 预热缓存（localStorage 已学到速率/字数比）下的开播时刻 */
const path = require("path");
const fs = require("fs");
const puppeteer = require("/Users/junjian/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core");

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const URL = process.env.APP_URL || "http://127.0.0.1:8817";
const OUT = path.join(__dirname, "shots");
const TEXT =
  "包含分段堆叠图、逐 block 边际成本柱图、四个证伪的原始数据、分层优化表和复现命令，" +
  "这些内容共同构成了这份报告的完整结构，也决定了它的阅读节奏。";

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required", "--force-device-scale-factor=2"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 2 });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => m.type() === "error" && errors.push("console: " + m.text()));

  await page.goto(`${URL}/?v=${Date.now()}`, { waitUntil: "networkidle2", timeout: 30000 });
  // 预置「第二段起」的缓存：上一段已学到 r=0.70、字数比 0.175
  await page.evaluate(() => {
    localStorage.setItem(
      "voxcpm2.streamRate",
      JSON.stringify({ gguf: { r: 0.7, spc: 0.175 }, mlx: { r: 1.39, spc: 0.18 } })
    );
    localStorage.removeItem("voxcpm2.lowLatency");
  });
  await page.reload({ waitUntil: "networkidle2" });
  await page.waitForFunction(
    () => document.getElementById("statusModel")?.textContent.includes("就绪"),
    { timeout: 150000 }
  );

  await page.evaluate(() => {
    document.querySelector('#engineSeg button[data-engine="gguf"]').click();
  });
  await new Promise((r) => setTimeout(r, 300));
  await page.evaluate((t) => {
    const el = document.getElementById("text");
    el.value = t;
    el.dispatchEvent(new Event("input"));
    document.getElementById("streamToggle").checked = true;
    window.__streamStats = null;
  }, TEXT);

  await page.click("#genBtn");
  await new Promise((r) => setTimeout(r, 4000));
  const mid = await page.evaluate(() => ({
    bufVisible: !document.getElementById("mBuf").hidden,
    bufText: document.getElementById("mBuf").textContent,
    dur: document.getElementById("mDur").textContent,
    overflow: document.querySelector(".meta").scrollWidth > document.querySelector(".meta").clientWidth,
    metaW: document.querySelector(".meta").scrollWidth + "/" + document.querySelector(".meta").clientWidth,
  }));
  await page.screenshot({ path: path.join(OUT, "stream-buffering.png") });

  await page.waitForFunction(() => !document.getElementById("genBtn").disabled, { timeout: 240000 });
  const stats = await page.evaluate(() => {
    const s = window.__streamStats || {};
    return {
      startedAtSec: s.startedAtSec,
      needSec: s.needSec,
      bufferedAtStart: s.bufferedAtStart,
      underrunCount: s.underrunCount,
      underrunMs: s.underrunMs,
      audioSec: s.audioSec,
      rateWall: s.rateWall,
    };
  });
  console.log(JSON.stringify({ mid, stats, errors }, null, 2));
  await browser.close();
})().catch((e) => {
  console.error("FAILED:", e.message);
  process.exit(1);
});
