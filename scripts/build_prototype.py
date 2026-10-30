"""Build web/prototype.html from web/prototype.src.html.

Copies in the recorded run (web/example.json), the recorded sweep
(web/sweep.json, if present) and the self-hosted fonts (web/fonts/), so the
built page opens from disk with no server and no network. Then, if Node and
Playwright are installed, it runs the page once and saves the drawn first
screen into the HTML, so previews that block scripts still show it.

Run:
    uv run python scripts/build_prototype.py [--no-prerender]
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
SRC = WEB / "prototype.src.html"
OUT = WEB / "prototype.html"

PRERENDER_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
  await p.goto('file://' + process.argv[1] + '?still', { waitUntil: 'load' });
  await p.waitForTimeout(400);
  process.stdout.write(await p.$eval('#shell', (e) => e.innerHTML));
  await b.close();
})();
"""


def embedded_fonts() -> str:
    """fonts.css with every url() replaced by a base64 data URI."""
    css = (WEB / "fonts" / "fonts.css").read_text()

    def data_uri(m: re.Match) -> str:
        font = (WEB / "fonts" / m.group(1)).read_bytes()
        return f"url(data:font/woff2;base64,{base64.b64encode(font).decode()})"

    return re.sub(r"url\('([^']+\.woff2)'\)", data_uri, css)


def prerender(page: Path) -> str | None:
    """The drawn #shell contents, or None when Node or Playwright is missing."""
    if not shutil.which("node"):
        return None
    run = subprocess.run(["node", "-e", PRERENDER_JS, str(page)],
                         capture_output=True, text=True, cwd=ROOT)
    return run.stdout if run.returncode == 0 and run.stdout else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-prerender", action="store_true",
                    help="skip saving the drawn first screen into the HTML")
    args = ap.parse_args()

    rec = json.loads((WEB / "example.json").read_text())
    sweep_path = WEB / "sweep.json"
    sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else None

    page = SRC.read_text()
    for marker in ("__REC__", "__SWEEP__", '<link rel="stylesheet" href="fonts/fonts.css">'):
        if page.count(marker) != 1:
            raise SystemExit(f"{SRC.name}: expected one {marker!r}")
    # json.dumps output is safe inside <script> once "</" is escaped.
    as_js = lambda v: json.dumps(v, ensure_ascii=False).replace("</", "<\\/")
    page = page.replace("__REC__", as_js(rec)).replace("__SWEEP__", as_js(sweep))
    page = page.replace('<link rel="stylesheet" href="fonts/fonts.css">',
                        "<style>\n" + embedded_fonts() + "</style>")
    OUT.write_text(page)

    drawn = None if args.no_prerender else prerender(OUT)
    if drawn:
        start = page.index('<div class="shell" id="shell">') + len('<div class="shell" id="shell">')
        end = page.index('</div>\n<div class="sr" id="live"')
        OUT.write_text(page[:start] + drawn + page[end:])
    runs = 1 + (len(sweep["runs"]) if sweep else 1)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB, "
          f"{runs} runs{', sweep' if sweep else ''}, "
          f"{'pre-rendered' if drawn else 'not pre-rendered: needs node + playwright'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
