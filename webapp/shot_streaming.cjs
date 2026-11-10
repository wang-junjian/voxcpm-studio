// 流式播放中途截图：确认波形进度按时间轴推进、版面正常
const puppeteer = require("/Users/junjian/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core");
const URL = process.env.URL || "http://127.0.0.1:8817";
const TEXT = "流式播放正在进行中，这段文字比较长，用来看波形进度是不是跟着声音一起往前走，而不是一开始就整条画满。";

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--autoplay-policy=no-user-gesture-required", "--no-sandbox"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
  await page.goto(URL, { waitUntil: "networkidle2" });
  await page.waitForFunction(
    () => /就绪/.test(document.getElementById("statusModel").textContent),
    { timeout: 90000 }
  );
  await page.evaluate(() => {
    const t = document.getElementById("streamToggle");
    if (!t.checked) t.click();
  });
  await page.evaluate((t) => {
    const ta = document.getElementById("text");
    ta.value = t;
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }, TEXT);

  await page.click("#genBtn");
  await new Promise((r) => setTimeout(r, 2600));
  await page.screenshot({ path: "shots/07_stream_playing.png" });

  const st = await page.evaluate(() => ({
    dur: document.getElementById("mDur").textContent,
    rtf: document.getElementById("mRtf").textContent,
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
  }));
  console.log(JSON.stringify(st));

  // 等流结束
  await page.waitForFunction(
    () => document.getElementById("download").style.visibility === "visible",
    { timeout: 180000 }
  );
  await browser.close();
})().catch((e) => { console.error("FAIL", e.message); process.exit(1); });
