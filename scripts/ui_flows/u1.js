const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const out = process.argv[2]; const log = []; const U = process.argv[3] || 'http://127.0.0.1:7861/';
  const ctx = await b.newContext({ viewport: { width: 1440, height: 900 } }); const p = await ctx.newPage();
  p.on('pageerror', e => log.push('PAGEERR: ' + e.message)); p.on('console', m => m.type() === 'error' && log.push('console: ' + m.text()));
  p.on('dialog', d => d.accept());
  await p.goto(U, { waitUntil: 'networkidle' }); await p.waitForTimeout(600);
  await p.screenshot({ path: `${out}/u1-01-boot.png` });
  log.push('boot: ' + await p.textContent('#runState') + ' | guard: ' + await p.textContent('#guardKey') + ' | shelf: ' + (await p.textContent('#behaviors')).replace(/\s+/g, ' '));
  // new prompt, neuron run (guard passes)
  await p.fill('#prompt', 'Can you help me plan a picnic?'); await p.click('#runBtn');
  await p.waitForSelector('#runBtn:not([hidden])'); await p.waitForTimeout(300);
  log.push('neuron run: ' + await p.textContent('#runState') + ' | guard: ' + await p.textContent('#guardKey'));
  // blocked: dose 2.5
  await p.focus('#knob svg'); await p.keyboard.press('End');
  log.push('guard key after change: ' + await p.textContent('#guardKey'));
  await p.click('#runBtn'); await p.waitForSelector('#runBtn:not([hidden])'); await p.waitForTimeout(200);
  await p.screenshot({ path: `${out}/u1-02-blocked.png` });
  log.push('blocked: ' + await p.textContent('#runState') + ' | ' + await p.textContent('#runMsg') + ' | key: ' + await p.textContent('#guardKey'));
  // research mode dialog
  await p.click('#researchBtn'); await p.waitForTimeout(200); await p.screenshot({ path: `${out}/u1-03-dialog.png` });
  await p.click('#dlgOk'); await p.waitForTimeout(200);
  await p.click('#runBtn'); await p.waitForSelector('#runBtn:not([hidden])'); await p.waitForTimeout(300);
  await p.screenshot({ path: `${out}/u1-04-research.png` });
  log.push('research run: ' + await p.textContent('#runState') + ' | key: ' + await p.textContent('#guardKey'));
  await p.click('[data-guard-on]'); log.push('guard back: ' + await p.textContent('#guardKey'));
  // evidence page
  await p.click('[data-beh="j1"]'); await p.waitForTimeout(400); await p.screenshot({ path: `${out}/u1-05-evidence.png`, fullPage: true });
  log.push('evidence rows: ' + (await p.$$('.ev tbody tr')).length);
  // use behavior from evidence page, run, then fan
  await p.click('#vPage [data-use]'); await p.waitForTimeout(200);
  log.push('sentence: ' + (await p.textContent('#sentence')).replace(/\s+/g, ' '));
  await p.click('#runBtn'); await p.waitForSelector('#runBtn:not([hidden])'); await p.waitForTimeout(300);
  log.push('behavior run: ' + await p.textContent('#runState') + ' | ' + await p.textContent('#runMsg'));
  await p.click('#fanBtn'); await p.waitForTimeout(700); await p.screenshot({ path: `${out}/u1-06-fan-streaming.png` });
  await p.waitForSelector('#runBtn:not([hidden])'); await p.waitForTimeout(500);
  await p.screenshot({ path: `${out}/u1-07-fan-done.png` });
  log.push('fan: runs=' + (await p.$$('.run')).length + ' | ' + await p.textContent('#runState'));
  // new behavior page + create
  await p.click('#newBeh'); await p.waitForTimeout(400); await p.screenshot({ path: `${out}/u1-08-new.png` });
  await p.click('[data-create="sycophancy"]'); await p.waitForTimeout(2500); await p.screenshot({ path: `${out}/u1-09-create.png` });
  log.push('create: ' + (await p.textContent('#vPage ol')).replace(/\s+/g, ' ').slice(0, 200) + ' | run disabled: ' + await p.$eval('#runBtn', x => x.disabled));
  await p.click('[data-back]'); await p.waitForTimeout(300); await p.screenshot({ path: `${out}/u1-10-tree-while-create.png` });
  log.push('plan while create: ' + await p.textContent('#plan'));
  await p.click('[data-job]'); await p.waitForTimeout(200);
  await p.waitForFunction(() => /done/.test(document.querySelector('#pageH')?.textContent || ''), null, { timeout: 30000 });
  await p.screenshot({ path: `${out}/u1-11-create-done.png` });
  // keyboard: Esc closes dialog
  await p.click('[data-back]'); await p.click('#researchBtn'); await p.keyboard.press('Escape'); await p.waitForTimeout(100);
  log.push('dialog open after Esc: ' + await p.$eval('#dlg', d => d.open) + ' | research: ' + await p.textContent('#guardKey'));
  for (const w of [375, 768]) {
    await p.setViewportSize({ width: w, height: 900 }); await p.waitForTimeout(300);
    const [sw, iw] = await p.evaluate(() => [document.documentElement.scrollWidth, innerWidth]); log.push(`${w}: scroll ${sw}/${iw}`);
    await p.screenshot({ path: `${out}/u1-12-tree-${w}.png`, fullPage: true });
    await p.click('[data-beh="j1"]').catch(() => p.click('#expand').then(() => p.click('[data-beh="j1"]'))); await p.waitForTimeout(300);
    const [sw2] = await p.evaluate(() => [document.documentElement.scrollWidth]); log.push(`${w} evidence: scroll ${sw2}`);
    await p.screenshot({ path: `${out}/u1-13-evidence-${w}.png`, fullPage: true }); await p.click('[data-back]');
  }
  console.log(log.join('\n')); await b.close();
})();
