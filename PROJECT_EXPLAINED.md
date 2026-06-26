# This project, explained from zero

You don't need to remember everything here today. This doc has three parts:
1. **What problem this solves** (plain English)
2. **What's actually happening** when you click "Run detection"
3. **CV bullet points** + a **1-month study plan** so you can defend this confidently in an interview

---

## Part 1: What problem is this even solving?

Imagine a factory with 5,000 machines, each with a vibration sensor bolted onto it, sending a reading 100 times per second. That's 500,000 numbers arriving every second, forever. Somewhere in that firehose of numbers, a bearing is starting to fail — its vibration pattern will look subtly "off" compared to its own recent history, hours before it actually breaks.

**The business problem:** if you can catch that "off" pattern in real time, you schedule maintenance before the machine fails (cheap, planned). If you can't keep up with the data and only check once an hour, you find out *after* it breaks (expensive, unplanned downtime). This is called **predictive maintenance**, and it's a real, named problem in manufacturing/industrial IoT.

**The technical problem:** checking "is this sensor's current reading weird compared to its recent history" is simple math (explained below), but doing it for 5,000+ sensors, continuously, fast enough to matter, is where it gets hard. A normal CPU does this work one-thing-at-a-time (or a handful of things at a time, with multi-core). A graphics card (GPU) — the same chip that renders video games — was actually originally built for a totally different reason but for the *same shape of problem*: doing the same small calculation thousands of times simultaneously. That's exactly what "check this sensor for anomalies" needs: the same calculation, repeated independently for every sensor.

**This project proves the GPU is actually faster for this, with real measured numbers** — not just "GPUs are fast" as a claim, but an actual stopwatch comparison.

---

## Part 2: What is "anomaly detection" math here, exactly?

Forget code for a second. Here's the actual math, in plain terms:

For one sensor, you have a stream of numbers over time: `..., 1.2, 0.9, 1.1, 1.3, 0.8, 5.0, ...`

To decide if `5.0` is "weird":
1. Look at the last 32 readings *before* it (this is the "window").
2. Calculate their **average** (mean) — say it's `1.05`.
3. Calculate how spread-out they normally are — the **standard deviation** — say it's `0.2`. (Std dev is just "on average, how far does each reading wander from the mean.")
4. Calculate: `(5.0 - 1.05) / 0.2 = 19.75`. This is called a **z-score** — "how many standard deviations away from normal is this point."
5. If that number is bigger than some threshold (we used `3`), flag it as an anomaly. `19.75 > 3`, so yes — `5.0` is wildly outside this sensor's normal behavior.

That's it. That's the entire algorithm: **rolling window → mean → standard deviation → z-score → compare to threshold.** "Rolling" just means you redo this for every single new point, always looking at the most recent 32 before it, like a sliding peephole moving through the data.

Now imagine doing that for 4,096 sensors, for 2,048 time points each, which is what the dashboard's default numbers (`4096` sensors, `2048` timesteps) mean. That's 4096 × 2048 ≈ 8.4 million individual z-score calculations. Still simple math, just a lot of repetitions.

---

## Part 3: What happens when you click "Run detection"

1. **The browser (frontend)** sends your numbers (4096 sensors, 2048 timesteps, window 32, threshold 3) to a server as a request — basically "hey, generate some fake sensor data with these settings and tell me which points look anomalous."

2. **The server (backend)** is a Python program (FastAPI) running on your laptop. It:
   - Makes up fake sensor data (random numbers, with a few obvious spikes planted in it on purpose, so we know some "real" anomalies exist to find)
   - Runs the rolling-window math above, for every sensor, using NumPy (a Python math library) — this is the "CPU path"
   - If a GPU were available, it would *also* run the exact same math on the GPU, in CUDA, and report that time too — that's the "GPU path." On your laptop right now, this shows "not available" because your laptop doesn't have the CUDA toolkit installed (we developed/tested the GPU code on a free cloud GPU called Google Colab instead — that part is proven to work, just not on this exact laptop).
   - Times both, packages up which sensors got flagged, and sends it all back as a JSON response.

3. **The browser** takes that response and draws it:
   - The bar chart = how many milliseconds the CPU took (and GPU, when available)
   - The line chart = one sensor's fake readings over time, with red dots wherever the math flagged a point as anomalous
   - The table = list of every sensor that had at least one anomaly, and how many

Nothing here is connected to a real factory — it's all **simulated data**, generated fresh every time you click the button, specifically so the project can be demoed without needing actual industrial sensors. The point being proven isn't "look at this real data," it's "look how much faster the GPU does the same math."

---

## Part 4: Why does the layout of data in memory matter? (the "deep" part)

This is the one technical concept this whole project is built to teach you deeply, so it's worth understanding even if everything else stays fuzzy.

A GPU doesn't run one calculation at a time like a CPU core does. It runs **groups of 32 calculations at the exact same instant**, called a "warp." Picture 32 workers on an assembly line, all reaching for their next part at the exact same moment, in lockstep.

Now: if those 32 workers' "parts" (memory addresses) are sitting right next to each other on a shelf, one delivery truck can grab all 32 in one trip. That's called **coalesced** memory access — efficient.

If those 32 workers' parts are scattered far apart from each other on different shelves, the delivery truck has to make 32 separate trips, one per worker, even though all 32 workers wanted their stuff at the exact same moment. That's **uncoalesced** access — wasteful, and this is exactly what slows the naive version of this project's kernel down.

**Concretely in this project:** we wrote the *exact same math* twice, with one difference — how the sensor data sits in memory:
- **Version 1 (naive):** each sensor's full history is stored together, back to back. Sounds organized — but when 32 sensors are all being checked simultaneously by 32 GPU workers, those 32 sensors' data is scattered far apart from each other in memory (since each sensor "owns" a big contiguous block). 32 separate trips needed.
- **Version 2 (the fix):** we flip the storage around so that "all sensors' value at this exact moment in time" are stored next to each other instead. Now when 32 GPU workers check the same moment in time simultaneously, their data is right next to each other. One trip.

We measured this with a stopwatch: **same answer, same algorithm, just rearranged storage — and version 2 was 14-20x faster.** That's the whole lesson: on a GPU, *how you arrange your data* can matter more than the math itself.

---

## CV bullet points (pick 2-3, don't use all of them — keep it tight)

- Built a CUDA-accelerated anomaly detection pipeline (rolling z-score) across thousands of simulated sensor streams, achieving up to 449x speedup over a vectorized NumPy baseline, validated bit-for-bit against the CPU reference at every tested scale.
- Diagnosed and fixed a memory-coalescing bottleneck in a custom CUDA kernel by restructuring the on-device data layout (sensor-major → time-major), measuring a 14-20x kernel-level speedup from that change alone.
- Built a full-stack demo (FastAPI + React/Vite) exposing the CUDA kernel via a pybind11 Python extension, with a live dashboard showing CPU-vs-GPU timing, flagged anomalies, and per-sensor signal visualization.
- Caught and fixed a memory-scaling bug in the CPU baseline (vectorized NumPy implementation OOM'd past ~16k sensors) by rewriting it with O(n) cumulative-sum math instead of materializing an O(n×window) tensor — re-verified correctness against a brute-force reference after the fix.

## One-line summary (for a CV header / portfolio card)

"CUDA-accelerated anomaly detector for industrial sensor streams — 449x faster than a vectorized NumPy baseline, with the GPU speedup traced to a measured memory-access-pattern fix."

---

## The 1-month "actually learn this" study plan

Goal: be able to explain every concept below in your own words, in an interview, without reading from notes. Don't try to learn CUDA from scratch in depth — focus on understanding *this project's* code well enough to defend it.

### Week 1: the math and the business problem (no code required)
- Re-read Part 1 and 2 of this doc until you can explain, out loud, to a friend with zero CS background: what predictive maintenance is, and what a rolling z-score is.
- Practice computing a z-score by hand on paper with 5 fake numbers. This sounds trivial but doing it once by hand makes the code make sense forever after.
- Look up (5-10 min, just enough to recognize the term in an interview): "predictive maintenance," "anomaly detection," "z-score / standard score."

### Week 2: how a GPU is different from a CPU (concepts, light code reading)
- Watch or read *one* beginner explainer on "what is a CUDA thread/block/grid" — you just need the mental model: a GPU runs thousands of tiny copies of the same function simultaneously, organized into groups of 32 ("warps") that move in lockstep.
- Open [cuda/kernels.cu](cuda/kernels.cu) and just read the comments — don't worry about syntax yet. Find the two functions `rollingZScore` and `rollingZScoreCoalesced`. Notice they do the *same* math, just index into the `data` array differently.
- Re-read Part 4 of this doc (the coalescing explanation) until the "32 workers reaching for parts on a shelf" analogy feels natural enough that you could explain it to someone else.

### Week 3: how the pieces connect (architecture)
- Open [README.md](README.md)'s "Structure" section and match each file to a stage: kernel math (`cuda/`) → Python wrapper that calls it (`cuda/bindings.cpp`, exposed as `anomaly_gpu` in Python) → web server that calls that (`backend/main.py`) → webpage that calls the web server (`frontend/src/App.jsx`).
- Open `backend/main.py` and find the `/run` endpoint. Read top to bottom — it's short. Match each line to a step in Part 3 of this doc ("generate fake data" → "run CPU math" → "run GPU math if available" → "package results").
- Click around the actual dashboard with different numbers (try `num_sensors: 100`) and watch how the table/chart change. Small numbers are easier to reason about than 4096.

### Week 4: numbers and defense practice
- Memorize (loosely, not word-for-word) the numbers in [README.md](README.md)'s Results table: speedup goes from ~57x at 1,024 sensors up to ~449x at 16,384 sensors. Be ready to say *why* it grows with scale (more parallel work for the GPU to do relative to its fixed overhead).
- Read [TALKING_POINTS.md](TALKING_POINTS.md) fully — it's written exactly for this moment. Practice the "30-second pitch" out loud 3-4 times until it doesn't feel memorized.
- Have someone (or yourself, recorded) ask you: "walk me through this project," "why is the GPU faster," "what would you do differently with more time," and answer out loud. The goal isn't a perfect script — it's being able to recover smoothly when you stumble.

### What you can honestly skip getting deep on
- Actual CUDA syntax (writing kernels from scratch) — you understand *this* kernel, that's enough for a first project.
- React/FastAPI deep internals — you can say "I used FastAPI for the backend API and React for the dashboard," and that's a completely normal, truthful level of depth for a first full-stack-plus-GPU project.
- Cloudflare Tunnel internals — "I exposed my local demo via a tunnel so I could share a live link without deploying to a cloud host" is enough.
