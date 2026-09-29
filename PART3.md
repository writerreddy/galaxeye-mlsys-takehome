# Part 3 — Problem-Solving

## Q1. The classifier is wrong ~30% of the time. What do you do, and how do you decide if it's good enough to be useful?

30% error rate by itself doesn't tell me whether the system is useful. It's
an average over things that matter a lot — which classes are getting
confused, how confident the model was when it was wrong, and what actually
happens downstream when a wrong label goes out.

So the first thing I'd do is stop looking at the single number and look at
the confusion matrix and per-class precision/recall. In the model I actually
built for this take-home, the aggregate accuracy on the held-out eval set is
about 85%, but that hides a lot: Forest, SeaLake, Industrial, and
Residential are all being classified correctly 93-98% of the time, while
Highway and River sit around 60-70% and get confused with each other a lot
(rivers and highways are both long, thin, linear features against a
background — it's an understandable mix-up, not a random failure). If
someone only cared about "is this a Forest," they'd have a genuinely
reliable system. If they cared about River vs. Highway specifically, they
would not.

Next I'd look at whether the model's confidence is actually informative —
i.e., is it telling the truth about when it's likely to be wrong? In this
system, yes, and by a lot: predictions at or above the 0.60 confidence
threshold are right about 97% of the time; predictions below it are right
only about half the time. That's the real lever. If confidence is
well-behaved like that, a 30% overall error rate can still be operationally
fine, because you're not treating all predictions equally — you route the
confident 70% one way and the uncertain 30% another way (human review, a
"needs verification" queue, excluded from any automated aggregate).

Then I'd compare against a trivial baseline (e.g., always predict the most
common class) — if the model isn't clearly beating that, it's not adding
value no matter how good it sounds. And I'd figure out, concretely, what a
wrong answer costs. "Wrong 30% of the time" is a very different problem if
the output feeds a fully automated pipeline than if it's advisory context
next to a human analyst. Without knowing that cost, I genuinely can't say
whether 30% is fine.

Practically, my sequence would be: (1) get the confusion matrix and
per-class numbers, not just the headline accuracy; (2) check whether
confidence correlates with correctness, and if so, use it as a real
routing signal, not just a display field; (3) find out from whoever owns
the downstream use case what an acceptable error rate actually is for their
specific decision; (4) if the model is genuinely too weak for the riskiest
classes, either improve the data/features for those classes specifically,
route them to mandatory human review, or explicitly scope them out; (5)
once deployed, keep watching the confidence and class distributions over
time (see Q2) since "good enough today" isn't a permanent property.

## Q2. This runs offline, no one watching live. A month later, how do you know it's still working?

Without live monitoring and without new ground-truth labels, I can't
directly measure "is accuracy still 85%." What I *can* do is watch for
signals that something has clearly gone wrong, using only what the system
already produces.

First, the boring infrastructure checks: is the process actually still
running, is the disk that holds the SQLite file getting full, are recent
rows even present (a service that's silently stopped receiving tiles looks
very different in the data from one that's still working). Those are cheap
and catch the most common failure mode — the thing just stopped, or
something upstream stopped feeding it.

Second, I'd pull distributional stats out of `classification_results`
directly, since every prediction is already logged there with its
confidence and class: prediction volume over time, the mix of predicted
classes, and the confidence distribution. If a month ago 15% of tiles came
back "uncertain" and now it's 60%, something changed — either the incoming
tiles look different than training data (drift), or something upstream is
now sending corrupted/different images. Similarly, if the class mix
suddenly shifts hard toward one class, that's worth a look even without
knowing if any individual prediction is "correct."

Third, I'd keep a small, fixed "golden set" of tiles with known correct
answers — tiles I'm confident about, not from the eval set used for
reporting — and periodically (say, weekly, via a cron-style script) run
them back through the exact same model artifact and confirm the predictions
haven't changed. This doesn't tell you about real-world accuracy, but it
catches "the model file got corrupted," "someone swapped in the wrong
artifact," or "a dependency upgrade silently changed behavior" — all real
risks for something running unattended for a month.

Fourth, I'd checksum the model artifact itself and log/verify it at
startup, and make sure `model_version` is stamped on every row, so if
someone *did* touch the model, it's visible in the data going forward
rather than silently blended in.

I want to be honest about the limit here: none of this proves the model is
still 85% accurate in the real world. Without labels, true accuracy can't
be automatically verified — only proxies for "has anything obviously
changed" can be. If real ongoing accuracy monitoring matters, the actual
fix is a process, not more code: get some fraction of predictions reviewed
by a person periodically and feed that back in as fresh ground truth.

## Q3. Tiles arrive fine, but stored results look wrong. Walk through your debugging steps.

In order:

1. **Reproduce it.** Get the exact tile (or a copy of it) and the exact
   stored result that looked wrong. If I can't reproduce it at all, I need
   more information before I can debug anything.
2. **Look at the raw input file itself** — open the image directly, outside
   the app. Is it actually the image I think it is? (Wrong file attached,
   truncated upload, wrong content-type, etc. — surprisingly common causes.)
3. **Confirm ingestion succeeded cleanly** — check the request logs for that
   tile: correct filename, correct size, no validation warnings.
4. **Verify decoding** — load the exact same bytes through
   `load_image_from_bytes` in isolation (a quick Python shell) and check the
   resulting image's size/mode look sane, not some garbled or wrongly
   color-shifted decode.
5. **Verify preprocessing** — run `extract_features` on that decoded image
   directly and sanity-check the vector (right length, no NaNs, values in
   the expected 0-1 range).
6. **Run the model independently**, outside the API, on that exact feature
   vector, and see what it predicts and with what confidence.
7. **Check which model artifact is actually loaded** — path, version,
   timestamp — in case the running process has a stale or unexpected
   `models/land_use_classifier.joblib`.
8. **Compare the "run it myself" prediction from steps 4-6 against what the
   API actually returned.** If they match, the bug isn't in inference — it's
   somewhere between the model's output and the database. If they don't
   match, the bug is in decoding/preprocessing/model loading, and I stop
   here and look at those.
9. **If they matched:** inspect the code path that turns the prediction into
   the stored row (`classify_and_store`) — is the class label being
   swapped, off-by-one indexed against the wrong `classes` list, or is the
   confidence threshold logic assigning the wrong status?
10. **Inspect the actual database write/read** — query the row directly
    with `sqlite3`, not through the API, and check the raw values.
11. **Check types/serialization** — is a float becoming a string somewhere,
    is a class name getting truncated by a column length, is there a
    silent unicode/encoding issue in the filename.
12. **Cross-check the `image_hash`** stored against a hash I compute myself
    from the same file — if they don't match, I'm not even looking at the
    row I think I am, and the mixup is elsewhere (e.g., a duplicate
    `tile_id`, or a race in how the row was fetched).
13. **Check timestamps and ordering** — if there are several results for
    similar filenames, am I even looking at the row from the request I
    care about, or an earlier/later one?
14. **Isolate to one component with a minimal repro** — once I know roughly
    where it broke (decode vs. features vs. model vs. persistence vs.
    serialization), write the smallest possible script that reproduces just
    that step.
15. **Add a regression test** that encodes this exact failure (this tile,
    this expected behavior) so it can't silently reappear.
16. **Fix it, then re-run the full pipeline test and the new regression
    test** to confirm the fix actually holds end-to-end, not just in
    isolation.

The general principle: work from the outside in, and at each boundary
(input file → decoded image → feature vector → model output → database row
→ API response) check whether that boundary is where the value became
wrong, rather than guessing based on where I assume the bug probably is.

## Q4. What's the weakest part of your design, and what breaks first?

The honest answer is a combination of two things that are really the same
underlying weakness: **the model has no reliable way to detect that its own
inputs have changed out from under it, and there's no ground truth to catch
that once deployed.**

Concretely: my classifier was trained on a specific, fairly narrow dataset
(EuroSAT-style 64x64 Sentinel-2 tiles). If GalaxEye's real tiles differ in
resolution, sensor, color balance, geographic region, or land-use mix from
what it saw in training, accuracy could degrade substantially without any
error being thrown — the code path runs "successfully" and returns a
confident-looking wrong answer. This is the first thing that would break in
practice: a change upstream (new sensor, new region, a preprocessing tweak
elsewhere in the pipeline) that silently shifts the input distribution, with
nothing in this system positioned to notice until someone manually compares
predictions to reality.

Two secondary weaknesses I'd also flag, in order of how much I'd worry
about them next: SQLite's single-writer model would become a real
bottleneck if this needed genuine concurrent write throughput (e.g., many
tiles being classified in true parallel rather than the light,
occasional load this take-home assumes) — that's a straightforward swap to
Postgres if it comes to that, not a design flaw so much as a scope choice.
And the confidence threshold, while genuinely useful on this dataset (see
Q1), is a single global cutoff — it isn't calibrated per class, so a class
where the model is systematically overconfident (or underconfident) won't
be caught by one shared number.

If I were told this needs to scale past a take-home: I'd add the "golden
set" regression checks from Q2 as a required, scheduled part of the
deployment (not just a nice-to-have), get a process in place for
periodically injecting real reviewed labels back into monitoring, and only
then start worrying about swapping in a heavier model — since without
distribution monitoring, a better model just fails the same way, quietly.
