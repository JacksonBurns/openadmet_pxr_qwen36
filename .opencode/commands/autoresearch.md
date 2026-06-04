# autoresearch

Do your own research toward developing a machine learning model to predict activity against PXR.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context, including the description of the challenge.
   - `examples/` - example code for using the various machine learning tools available to you
   - `experiment.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Initialize lab_notebook.txt**: Create `lab_notebook.txt`.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

You will launch the experiment by simply running `python experiment.py`.

**What you CAN do:**
- Modify `experiment.py` — this is the only file you edit.
Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
You can change which targets are regressed, how you preprocess the molecular structures, which data you use, how you split, anything related to the process of delivering the machine learning model.

**What you CANNOT do:**
- Modify other files: only modify `experiment.py`, other files are irrelevant to you.
- Install new packages or add dependencies: you can only use `molpipeline`, `chemprop`, `xgboost`, and `scikit-learn`.
- Modify the evaluation data: the evaluation data is static - do not change, and do not use it for training or validation, only testing of your models.

**The goal is simple: get the lowest evaluation MAE.** Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.
Do not try and find a baseline starting point from the commit history or your context - ALWAYS run the `experiment.py` as is, and use the result as the baseline.

## Output format

Once the script finishes it writes to a file called `results.csv` that looks like this (this is just an example, do not actually use these numbers):

```
Timestamp,MAE,Execution Time
2026-06-03 23:32:26,0.5105,1079.5s
2026-06-04 01:06:24,0.4923,1072.9s
2026-06-04 01:26:50,"CRASH (AttentiveAggregation.__init__() missing 1 required keyword-only argument: 'output_size')",54.8s
```

The MAE for the most recent execution will be on the last line (`tail` and `grep`, for example, can retrieve it).
If the run crashed, MAE will instead say what happened with the word "CRASH".

## Lab Notebook

As you run experiments, you should maintain a virtual lab notebook in a text file (`lab_notebook.txt`).
Each line in `lab_notebook.txt` should correspond to a line in `results.csv`.
In this lab notebook you should record what happened - did the execution crash, and why?
How did the model performance change, and how did this compare to what you were expecting?
What is the next best model to try, and what evidence does this result provide to support or refute your model development hypothesis?

## Plateau detection

Track your recent improvement rate. If your last 5 consecutive experiments all failed to meaningfully improve, you have hit a **plateau**. When this happens:

- **Stop tuning hyperparameters.** More LR/batch-size/warmup sweeps will not help.
- **Make a structural change.** This means changing the model architecture itself: different attention mechanism, different normalization, different positional encoding, adding/removing layers, changing the optimizer algorithm, etc.
- **Try something you haven't tried before.** Re-read `experiment.py` from scratch for new angles. Consult the comments and references in the code for ideas from the literature.

The pattern is: early experiments should explore broadly across categories, middle experiments can tune what works, and when you plateau you must escape via a structural leap.

## Experiment categories

To maintain diversity, mentally categorize each experiment as one of:

- **Architecture**: layer count, width, attention type, normalization, positional encoding
- **Optimization**: optimizer, learning rate schedule, warmup, weight decay, gradient clipping
- **Training dynamics**: batch size, sequence packing, gradient accumulation, loss function
- **Simplification**: removing components, reducing complexity for equal or better results

Avoid running more than 3 experiments in the same category in a row. If you've been tuning LR for 3 runs, switch to an architecture change. Diversity of exploration beats depth of exploitation in the early/mid phase.

## Prior Knowledge

Below are some observations from previous iterations of research, which you should incorporate into your design:

 - The CheMeleon foundation model is a strong model which should always be incorporated into the ensemble
 - Random SMILES augmentation has no impact on model performance, since neither fingerprint nor Chemprop act on the SMILES directly

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `experiment.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python experiment.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results from `results.csv`
6. If the the run crashed, run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in in your lab notebook, and make to include these in git.
8. If improved (lower), you "advance" the branch, keeping the git commit
9. If equal or worse, you revert the commit to preserve the attempt and then proceed with a new idea.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
