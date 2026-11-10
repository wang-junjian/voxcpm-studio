// 连续两次流式生成 + 流尾回放：确认无异常、无叠音残留
const puppeteer = require("/Users/junjian/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core");
const URL = process.env.URL || "http://127.0.0.1:8817";

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--autoplay-policy=no-user-gesture-required", "--no-sandbox"],
  });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });

  await page.goto(URL, { waitUntil: "networkidle2" });
  await page.waitForFunction(
    () => /就绪/.test(document.getElementById("statusModel").textContent),
    { timeout: 90000 }
  );
  await page.evaluate(() => {
    const t = document.getElementById("streamToggle");
    if (!t.checked) t.click();
  });

  const run = async (text, waitBlob = true) => {
    await page.evaluate((t) => {
      const ta = document.getElementById("text");
      ta.value = t;
      ta.dispatchEvent(new Event("input", { bubbles: true }));
    }, text);
    await page.evaluate(() => {
      document.getElementById("download").style.visibility = "hidden";
    });
    await page.click("#genBtn");
    if (waitBlob) {
      await page.waitForFunction(
        () => document.getElementById("download").style.visibility === "visible",
        { timeout: 180000 }
      );
    }
  };

  // 第一段：等 blob 就绪后立刻发起第二段（此时尾巴大概率还在播）
  await run("第一段测试，尾巴还在播放的时候立刻发起第二段。");
  await run("第二段测试，用来确认上一段残留的音源已经被停掉，不会叠在一起。");

  // 流尾仍在播时点“播放”回放整段
  await page.click("#playBtn");
  await new Promise((r) => setTimeout(r, 1200));

  const st = await page.evaluate(() => ({
    playing: !window.__vox.audio.paused,
    dur: window.__vox.audio.duration,
    ctx: window.__vox.state.duration,
  }));

  // 第三段：回放中再生成，确认不冲突
  await run("第三段测试，在回放过程中再次生成，验证切换是否干净。");

  console.log(JSON.stringify({
    afterReplay: st,
    errors,
  }, null, 2));
  await browser.close();
})().catch((e) => { console.error("FAIL", e.message); process.exit(1); });
