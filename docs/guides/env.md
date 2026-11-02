# Set up `.env`

`.env` holds local settings and secrets, for example `META_API_KEY`.
lobot reads it when a program starts, so you do not export variables in
every shell.

Warnings first:
- Never commit `.env`. Git ignores it. If `git check-ignore .env`
  prints nothing on your branch, merge `develop` first.
- Never paste the key into chat, a commit, a PR or a log.
- A variable set in your shell wins over `.env`. If the server uses the
  wrong key, run `unset META_API_KEY` and start it again.

## What reads `.env`

| Program | Reads `.env` |
|---|---|
| The server (`python -m model_lobotomy.server`) | Yes, at start, before any setting |
| `scripts/agent_create.py` | Yes |
| `scripts/validate_judge.py` | Yes |
| Other scripts and the unit tests | No. Export the variable in the shell |

The tests never read `.env`, so your key cannot change a test result.

## Set it up

1. Go to the repo root.
2. Check that git ignores `.env`:

   ```sh
   git check-ignore .env && echo "ignored: safe"
   ```

3. Copy the example file:

   ```sh
   cp .env.example .env
   ```

4. Put your Meta API key in `.env`. If you keep the key in a file, copy it
   without printing it (this replaces the file content):

   ```sh
   printf 'META_API_KEY=%s\n' "$(tr -d '\n' < ~/.config/lobot/meta_api_key)" > .env
   ```

   Or open `.env` in an editor and set `META_API_KEY=<your key>`.
5. Make the file private:

   ```sh
   chmod 600 .env
   ```

6. Optional: add other settings, one per line. For example, the llama lane
   for Create from the app:

   ```sh
   echo 'LOBOT_CREATE_LANE=llama' >> .env
   ```

   Every variable is in [configuration.md](../reference/configuration.md).

## File format

- One `NAME=value` per line.
- Lines that start with `#` are comments. Empty lines are skipped.
- `export NAME=value` also works.
- Quotes around the value are removed: `NAME="value"` and `NAME='value'`
  give `value`.
- No variable expansion: `$HOME` stays `$HOME`.

## Check that it works

1. Start the server:

   ```sh
   uv run python -m model_lobotomy.server
   ```

2. Read the start log. With the key loaded, it says:

   ```
   [server] Meta API: on, muse-spark-1.3 (from .env)
   ```

   Without a key, it says `Meta API: off (set META_API_KEY in .env)`.
3. Optional: ask the server. `meta_api.key` is `true` when the key loaded.
   The server never sends the key itself.

   ```sh
   curl -s http://127.0.0.1:7860/api/state | python3 -c "import json,sys; print(json.load(sys.stdin)['meta_api'])"
   ```

## What the key turns on

- **In plain words** on every evidence page: Muse Spark writes a short
  summary of the report as the page opens. The server keeps one summary
  per report.
- **Let Muse build it** on the New behavior page (the Muse agent).
- **Ask Muse for suggestions** on the labeling page.
- `scripts/agent_create.py` and `validate_judge.py --judge meta_api`.

Without the key, everything else works. These features show what is
missing, and their routes return 503.

## If it does not work

- **The log says `off`:** check that `.env` is at the repo root, not in
  `src/` or `docs/`. Check the name is exactly `META_API_KEY` and the line
  has an `=`.
- **The log says `on` but not `(from .env)`:** the shell has
  `META_API_KEY` set. The shell value wins. Run `unset META_API_KEY` to use
  `.env`.
- **Muse calls fail with an API error:** the key may be wrong or expired.
  Put the new key in `.env` and restart the server. The server reads
  `.env` only at start.
