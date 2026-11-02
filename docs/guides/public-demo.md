# Public demo (Hugging Face Space)

The public demo runs lobot in replay mode on a public host. It plays
recorded runs of Llama 3.2 3B Instruct. It needs no weights, no GPU and
no API key.

Warnings first:
- Never set `META_API_KEY` on the Space. The server refuses to start in
  public mode when the key is set.
- The demo shows the jobs in your local `jobs/` folder. Check that every
  job in it is one you want to publish.

## What the public mode does

`LOBOT_PUBLIC_DEMO=1` turns on these rules:

| Rule | Why |
|---|---|
| It needs `MODEL_LOBOTOMY_REPLAY=1` | No model loads, so no visitor can start a model job |
| It refuses to start with `META_API_KEY` set | No visitor can spend API credits |
| POSTs answer 403, except run, stop, chat replay, explain and second opinion | Visitors cannot write files or change shared state |
| Explain and second opinion return saved answers only | Muse Spark is never called |
| Same-origin POSTs pass on any host name | The Space has a real host name, not a loopback address |

`LOBOT_MODEL=meta-llama/Llama-3.2-3B-Instruct` starts the demo on Llama.

## Build the Space folder

1. Go to the repo root.
2. Build the folder from the code at `HEAD` and the jobs in `jobs/`:

   ```sh
   deploy/space/build.sh /tmp/lobot-space
   ```

   The script copies the code with `git archive`, so `.env` never enters
   the folder. It rewrites local paths in the jobs to `/app`. It stops if a
   local path or your key value is still in the folder.

## Test it locally

1. Build the image:

   ```sh
   docker build --platform linux/amd64 -t lobot-space /tmp/lobot-space
   ```

2. Run it:

   ```sh
   docker run --rm -p 7865:7860 lobot-space
   ```

3. Open `http://127.0.0.1:7865`. The start log says `REPLAY set` and
   `Meta API: off`.

## Deploy

1. Sign in once: `uv run hf auth login`.
2. Create a Docker Space on huggingface.co, for example `<user>/lobot`.
3. Upload the folder:

   ```sh
   uv run hf upload <user>/lobot /tmp/lobot-space . --repo-type space
   ```

4. Wait for the Space to build. The first build takes about 10 minutes.

To update the demo, build the folder again and upload it again.
