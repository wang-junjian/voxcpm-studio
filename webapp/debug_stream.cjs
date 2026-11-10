/* 定位流式请求的 ERR_ABORTED 来源 */
const puppeteer = require("/Users/junjian/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core");

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required", "--mute-audio"],
  });
  const page = await browser.newPage();
  const t0 = Date.now();
  const log = [];
  page.on("request", (r) => { if (r.url().includes("stream")) log.push(`[${Date.now()-t0}ms] request ${r.method()} ${r.url()}`); });
  page.on("requestfinished", (r) => { if (r.url().includes("stream")) log.push(`[${Date.now()-t0}ms] finished ${r.url()}`); });
  page.on("requestfailed", (r) => log.push(`[${Date.now()-t0}ms] FAILED ${r.url()} ${r.failure()?.errorText}`));

  await page.goto("http://127.0.0.1:8817", { waitUntil: "networkidle2" });
  await page.waitForFunction(() => document.getElementById("statusModel")?.textContent.includes("就绪"), { timeout: 60000 });

  // 纯 fetch 读流，不碰 AudioContext，看是否还 abort
  const r1 = await page.evaluate(async () => {
    const t = performance.now();
    const res = await fetch("/api/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "纯流式读取测试", mode: "basic", cfg: 2, steps: 10, seed: 1 }),
    });
    const reader = res.body.getReader();
    let bytes = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.length;
    }
    return { bytes, ms: Math.round(performance.now() - t) };
  });
  log.push(`[${Date.now()-t0}ms] 纯读取完成 ${JSON.stringify(r1)}`);

  await new Promise((r) => setTimeout(r, 2000));
  log.push(`[${Date.now()-t0}ms] 等待结束，准备关闭`);

  console.log(log.join("\n"));
  await browser.close();
})();
