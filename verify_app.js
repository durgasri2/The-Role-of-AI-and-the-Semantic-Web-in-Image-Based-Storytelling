const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  try {
    await page.goto('http://127.0.0.1:7860', { waitUntil: 'networkidle', timeout: 30000 });
    console.log('Page title:', await page.title());
    await page.screenshot({ path: 'screenshot.png', fullPage: true });
    console.log('Screenshot saved to screenshot.png');
  } catch (error) {
    console.error('Error visiting page:', error.message);
  } finally {
    await browser.close();
  }
})();
