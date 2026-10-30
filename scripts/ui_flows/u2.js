const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const out = 'u2'; const log = []; const U = 'http://127.0.0.1:7861/';
  const ctx = await b.newContext({ viewport: { width: 1440, height: 900 } }); const p = await ctx.newPage();
  p.on('pageerror', e => log.push('PAGEERR: ' + e.message));
  await p.goto(U, { waitUntil: 'networkidle' }); await p.waitForTimeout(500);
  await p.fill('#prompt', 'Tell me a story about the sea.'); await p.click('#runBtn'); await p.waitForSelector('#runBtn:not([hidden])'); await p.waitForTimeout(300);
  // click a word in the edited run
  await p.click('.run[aria-current="true"] .w >> nth=1'); await p.waitForTimeout(600);
  await p.screenshot({ path: `${out}/u2-1-word.png` });
  log.push('list items: ' + (await p.$$('#inspector li')).length);
  // open card
  await p.click('#inspector [data-card] >> nth=0'); await p.waitForTimeout(400);
  await p.screenshot({ path: `${out}/u2-2-card.png` });
  log.push('card: ' + (await p.textContent('#inspector')).replace(/\s+/g, ' ').slice(0, 200));
  // push from card (L11N1343 is already in edits from preset? then push 2nd)
  await p.keyboard.press('Escape'); await p.waitForTimeout(100);
  const pushes = await p.$$('#inspector [data-push]:not([disabled])'); log.push('push buttons: ' + pushes.length);
  await pushes[0].click(); await p.waitForTimeout(100);
  log.push('sentence: ' + (await p.textContent('#sentence')).replace(/\s+/g, ' '));
  await p.click('#runBtn'); await p.waitForSelector('#runBtn:not([hidden])'); await p.waitForTimeout(900);
  log.push('after run: ' + await p.textContent('#runMsg'));
  await p.screenshot({ path: `${out}/u2-3-proof.png` });
  // find
  await p.keyboard.press('Escape');
  await p.fill('#findIn', 'pirate'); await p.click('#findForm button'); await p.waitForTimeout(300);
  await p.click('#findOut [data-card] >> nth=2'); await p.waitForTimeout(400);
  await p.screenshot({ path: `${out}/u2-4-find.png` });
  log.push('find card: ' + (await p.textContent('#iHead')));
  // unreadable layer
  await p.click('#findOut [data-card] >> nth=2'); await p.waitForTimeout(300);
  for (const w of [375, 768]) {
    await p.setViewportSize({ width: w, height: 900 }); await p.waitForTimeout(300);
    const sw = await p.evaluate(() => document.documentElement.scrollWidth); log.push(`${w}: scroll ${sw}`);
    await p.screenshot({ path: `${out}/u2-5-${w}.png`, fullPage: true });
  }
  console.log(log.join('\n')); await b.close();
})();
