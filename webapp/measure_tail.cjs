// 验证流式播放不再截尾：
//  1) AudioContext 是否被 close（修复后应为 0 次）
//  2) 播放时间轴是否真的走到最后一块音频的结束点（scheduledEnd）
//  3) 期间是否有音源被 stop() 提前掐断
const puppeteer = require("/Users/junjian/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core");

const URL = process.env.URL || "http://127.0.0.1:8817";
const TEXT =
  process.env.TEXT ||
  "这句话的最后几个字必须在流式播放里完整念出来，不能丢尾巴。";

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--autoplay-policy=no-user-gesture-required", "--no-sandbox"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  await page.evaluateOnNewDocument(() => {
    window.__sched = [];
    window.__closes = 0;
    window.__forcedStops = 0;
    window.__lastAC = null;

    const OrigAC = window.AudioContext || window.webkitAudioContext;
    function PatchedAC(...args) {
      const ac = new OrigAC(...args);
      window.__lastAC = ac;
      const origCreate = ac.createBufferSource.bind(ac);
      ac.createBufferSource = function () {
        const src = origCreate();
        const origStart = src.start.bind(src);
        const origStop = src.stop.bind(src);
        let endedNaturally = false;
        src.addEventListener("ended", () => { endedNaturally = true; });
        src.start = function (when) {
          window.__sched.push({
            start: typeof when === "number" ? when : ac.currentTime,
            dur: src.buffer ? src.buffer.duration : 0,
          });
          return origStart(when);
        };
        src.stop = function (...a) {
          if (!endedNaturally) window.__forcedStops++;
          return origStop(...a);
        };
        return src;
      };
      const origClose = ac.close.bind(ac);
      ac.close = function () {
        window.__closes++;
        return origClose();
      };
      return ac;
    }
    PatchedAC.prototype = OrigAC.prototype;
    window.AudioContext = PatchedAC;
    window.webkitAudioContext = PatchedAC;
  });

  await page.goto(URL, { waitUntil: "networkidle2" });
  await page.waitForFunction(
    () => document.getElementById("statusModel") &&
          /就绪/.test(document.getElementById("statusModel").textContent),
    { timeout: 90000 }
  );

  const streamOn = await page.evaluate(() => {
    const t = document.getElementById("streamToggle");
    if (!t.checked) t.click();
    return t.checked;
  });

  await page.evaluate((t) => {
    const ta = document.getElementById("text");
    ta.value = t;
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }, TEXT);

  await page.click("#genBtn");
  await page.waitForFunction(
    () => document.getElementById("download").style.visibility === "visible",
    { timeout: 180000 }
  );

  const endInfo = await page.evaluate(() => {
    const s = window.__sched;
    const lastEnd = s.reduce((m, x) => Math.max(m, x.start + x.dur), 0);
    return {
      chunks: s.length,
      lastEnd,
      scheduledAudio: s.reduce((n, x) => n + x.dur, 0),
      ctxNow: window.__lastAC ? window.__lastAC.currentTime : null,
      closes: window.__closes,
      blobDuration: window.__vox.state.duration,
    };
  });

  // 等到播放时间轴越过最后一块的结束点（尾部真正播完）
  const reached = await page
    .waitForFunction(
      (t) => window.__lastAC && window.__lastAC.currentTime >= t,
      { timeout: 20000 },
      endInfo.lastEnd
    )
    .then(() => true)
    .catch(() => false);

  const after = await page.evaluate(() => ({
    ctxNow: window.__lastAC ? window.__lastAC.currentTime : null,
    state: window.__lastAC ? window.__lastAC.state : null,
    closes: window.__closes,
    forcedStops: window.__forcedStops,
  }));

  const gap = endInfo.lastEnd - endInfo.scheduledAudio;   // 首块起始偏移
  console.log(JSON.stringify({
    streamOn,
    chunks: endInfo.chunks,
    scheduledAudio: +endInfo.scheduledAudio.toFixed(3),
    blobDuration: +endInfo.blobDuration.toFixed(3),
    audioMatch: Math.abs(endInfo.scheduledAudio - endInfo.blobDuration) < 0.05,
    firstChunkOffset: +gap.toFixed(3),
    lastEnd: +endInfo.lastEnd.toFixed(3),
    ctxAtBlobReady: +endInfo.ctxNow.toFixed(3),
    tailReached: reached,
    ctxStateAtEnd: after.state,
    closes: after.closes,
    forcedStops: after.forcedStops,
    verdict:
      reached && after.closes === 0 && after.forcedStops === 0
        ? "PASS 尾部完整播放"
        : "FAIL 仍有截断",
  }, null, 2));

  await browser.close();
})().catch((e) => {
  console.error("FAIL", e.message);
  process.exit(1);
});
