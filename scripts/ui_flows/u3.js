const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const log = []; const p = await (await b.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
  p.on('pageerror', e => log.push('PAGEERR: ' + e.message));
  await p.goto('http://127.0.0.1:7861/', { waitUntil: 'networkidle' }); await p.waitForTimeout(500);
  await p.fill('#prompt', 'What do you think of my essay?');
  await p.click('#readBtn'); await p.click('[data-rd="hostility"]');
  log.push('sentence: ' + (await p.textContent('#sentence')).replace(/\s+/g, ' '));
  await p.focus('#knob svg'); await p.keyboard.press('End'); await p.keyboard.press('PageDown'); await p.keyboard.press('PageDown');
  await p.click('#runBtn'); await p.waitForSelector('#runBtn:not([hidden])'); await p.waitForTimeout(700);
  await p.screenshot({ path: 'u2/u3-1-dials.png' });
  await p.click('.run[aria-current="true"] .w >> nth=4'); await p.waitForTimeout(500);
  log.push('inspector: ' + (await p.textContent('#inspector .kv')).replace(/\s+/g, ' '));
  await p.screenshot({ path: 'u2/u3-2-inspect.png' });
  await p.setViewportSize({ width: 375, height: 900 }); await p.waitForTimeout(300);
  log.push('375 scroll ' + await p.evaluate(() => document.documentElement.scrollWidth));
  await p.screenshot({ path: 'u2/u3-3-375.png', fullPage: true });
  console.log(log.join('\n')); await b.close();
})();
