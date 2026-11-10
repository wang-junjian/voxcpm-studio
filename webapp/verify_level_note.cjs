/* 验证「换种子重试 / 响度归一」提示：页面上能正确显示，且不撑破结果栏 */
const path = require("path");
const fs = require("fs");
const puppeteer = require("/Users/junjian/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const URL = process.env.APP_URL || "http://127.0.0.1:8817";
const OUT = path.join(__dirname, "shots");

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required", "--mute-audio"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 2 });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => m.type() === "error" && errors.push("console: " + m.text()));

  await page.goto(URL, { waitUntil: "networkidle2" });

  // 选 GGUF、填入已知会退化的文本与 seed=42
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll("[data-engine]")].find(
      (b) => b.dataset.engine === "gguf"
    );
    if (btn) btn.click();
  });
  await page.evaluate(() => {
    document.querySelector("#text").value =
      "包含分段堆叠图、逐 block 边际成本柱图、四个证伪的原始数据、分层优化表和复现命令。";
    document.querySelector("#seed").value = "42";
    document.querySelector("#steps").value = "10";
    document.querySelector("#stepsVal").textContent = "10";
  });

  await page.click("#genBtn");
  await page.waitForFunction(
    () => !document.querySelector("#result").hidden,
    { timeout: 180000 }
  );
  await new Promise((r) => setTimeout(r, 800));

  const info = await page.evaluate(() => {
    const note = document.querySelector("#mNote");
    const meta = document.querySelector(".meta");
    const res = document.querySelector("#result");
    return {
      noteHidden: note.hidden,
      noteText: note.textContent,
      metaWidth: Math.round(meta.getBoundingClientRect().width),
      resultWidth: Math.round(res.getBoundingClientRect().width),
      overflow: meta.scrollWidth > meta.clientWidth + 1,
      engine: document.querySelector("#mEngine").textContent,
      dur: document.querySelector("#mDur").textContent,
      rtf: document.querySelector("#mRtf").textContent,
    };
  });

  const el = await page.$("#result");
  await el.screenshot({ path: path.join(OUT, "level-note.png") });

  console.log(JSON.stringify({ ...info, errors, ok: errors.length === 0 }, null, 2));
  await browser.close();
})();
