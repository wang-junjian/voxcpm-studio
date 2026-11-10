/* 移动端布局检查 + 窄屏截图 */
const puppeteer = require("/Users/junjian/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core");

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--no-sandbox", "--mute-audio"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await page.goto("http://127.0.0.1:8817", { waitUntil: "networkidle2" });
  await page.waitForFunction(
    () => document.getElementById("statusModel")?.textContent.includes("就绪"),
    { timeout: 60000 }
  );
  const check = await page.evaluate(() => {
    const docW = document.documentElement.clientWidth;
    const bad = [];
    document.querySelectorAll("*").forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.width > docW + 2) bad.push(`${el.tagName}.${el.className}`.slice(0, 50));
    });
    return {
      docW,
      scrollW: document.documentElement.scrollWidth,
      horizontalOverflow: bad.slice(0, 6),
      hScroll: document.documentElement.scrollWidth > docW + 1,
    };
  });
  await page.screenshot({ path: __dirname + "/shots/06_mobile.png", fullPage: true });
  console.log(JSON.stringify(check, null, 2));
  await browser.close();
})();
