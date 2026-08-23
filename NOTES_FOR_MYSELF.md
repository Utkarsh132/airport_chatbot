# Project Status Notes(for myself)

Last updated: August 23, 2026, 11:58 PM CEST

This file summarizes what has been fixed, tested, and confirmed so far, plus
known limitations worth mentioning in the assignment report.

---

## Fixes applied so far

### 1. Typo tolerance (text pipeline)
- Added typo correction (`correct_typos`) and fuzzy fallback intent matching
  in `src/text_pipeline.py`.
- Confirmed working live: "whre is gaet b12" correctly resolves to Gate B12
  with confidence 1.00.

### 2. Score normalization bug (fusion engine)
- `src/utils.py` -> `normalize_score()` originally assumed similarity scores
  range from -1 to +1 (raw cosine similarity range).
- In reality, TF-IDF cosine similarity and the image histogram similarity
  used in this project are already non-negative (0 to 1).
- This bug was inflating weak/unrelated matches -- e.g. a real similarity of
  0.18 was being displayed as 0.59, pushing irrelevant matches above the
  "uncertain" confidence threshold.
- Fixed by changing the default range in `normalize_score()` to [0.0, 1.0].

### 3. Flight-status override (fusion engine)
- `src/fusion.py` now explicitly detects `flight_status` intent and always
  returns the proper "I can't check live flight status" redirect message,
  overriding any KB match -- even a high-scoring one.
- This matches the project's own design decision: flight status requires
  live airline data that is not available in the static knowledge base.

### 4. Real Whisper transcription enabled
- `src/config.py` -> `USE_WHISPER` flipped from `False` to `True`.
- Previously the app used a stub transcriber that just looked up
  ground-truth text by filename -- it never actually processed real audio.
- Generated 5 real synthetic `.wav` files using macOS `say` (text-to-speech)
  based on the existing `data/audio/audio_metadata.json` ground-truth
  transcripts, using different voices/accents/speeds to simulate the
  documented test conditions (clear speech, background noise, non-native
  accent, fast speech, phone mic).
- Ran real faster-whisper ("tiny" model) transcription on these files.
  Result: average Word Error Rate (WER) = 0.057 across 5 samples.
  Only real error: the non-native-accent sample ("Is there a lounge near
  terminal 2?") was transcribed as "is their lounge near terminal 2." --
  a classic there/their homophone confusion, a known accent-related
  Whisper error type.
- Confirmed working live in the browser: voice-only queries correctly
  transcribe, route through the same NLP pipeline as typed text, and
  retrieve the correct knowledge base record.

### 5. Unified single chat bar (Streamlit UI)
- Rebuilt `app/streamlit_app.py` to use one input row: an attach button
  (accepting image OR audio via `st.popover`), a text box, and a Send
  button -- replacing the old layout of separate uploaders and dropdowns.
- Conversation now renders as chat bubbles with history.

---

## What has been tested live and confirmed working

| Scenario | Result |
|---|---|
| Text only (typo-tolerant) | Correct, high confidence |
| Voice only (clear speech) | Correct transcription + correct answer, confidence 1.00 |
| Voice only (accented speech, degraded transcript) | Correct answer despite transcription errors, confidence honestly dropped to 0.57 |
| Out-of-domain text ("tell me a joke") | Correctly triggered uncertain response, confidence 0.00 |
| Image only (synthetic dataset icon) | Correct, confidence 1.00 (memorized match, expected) |
| Image only (real-world photo, novel/dissimilar) | Correct category, confidence 0.55 (genuine CLIP generalization) |

## Added: camera button and mic button in the UI

- The attach popover now has 3 tabs: Upload (existing file uploader),
  Camera, and Mic.
- Camera tab uses `st.camera_input(...)` -- lets you take a live photo
  with your webcam/laptop camera to search against the knowledge base,
  instead of only uploading a pre-existing image file. Works fine on the
  installed Streamlit 1.35.0.
- Mic tab uses `st.audio_input(...)` for live voice recording, but this
  widget requires Streamlit >= 1.36. The installed environment is
  currently on 1.35.0, and `pip install --upgrade streamlit` could not
  fetch a newer version because network/DNS access was unreliable in this
  environment during setup.
- To avoid breaking the app, the Mic tab checks `hasattr(st,
  "audio_input")` at runtime: if unavailable (as on 1.35.0), it shows an
  info message and tells the user to use the Upload tab with a
  .wav/.mp3/.m4a file instead, rather than crashing.
- **RESOLVED**: Streamlit did silently upgrade to >=1.36 (pip upgrade
  appeared to already have the newer version available internally). The
  Mic tab initially showed "An error has occurred" -- this turned out to
  be a browser microphone permissions issue, not a code bug. Granting
  mic access in the browser fixed it immediately.
- **Confirmed working**: live mic recording via the Mic tab transcribed
  "Where is gate B12?" correctly with confidence 1.00, matching Gate B12
  in the knowledge base. Full end-to-end voice-via-mic pipeline is
  confirmed functional.
- File changed: `app/streamlit_app.py`. All three input sources (upload,
  camera, mic) feed into the same `attached_image_file` /
  `attached_audio_file` variables used by the rest of the app, so no
  changes were needed in `run_inference`/the fusion engine.

## Fixed: uploader only allowed one file at a time

- The attach button previously used `st.file_uploader(..., accept_multiple_files=False)`,
  which meant you could attach EITHER one image OR one audio clip, never both
  -- making the "voice + image combined" test impossible from the UI.
- Fix: changed to `accept_multiple_files=True`, then the app splits the
  uploaded files into at most one image + one audio file (warns if you
  attach more than one of the same type). `run_inference()` already
  supported passing both `image` and `audio_path` together, so no changes
  were needed in the fusion engine itself -- this was purely a UI
  limitation.
- File changed: `app/streamlit_app.py`.
- You can now attach both a photo and a `.wav`/`.mp3`/`.m4a` file in the
  same message to test voice + image combined.

## Voice + image combined -- tested, works

- Confirmed the uploader fix works: attached the real-world restroom
  photo ("WOMEN/Wahine" sign) together with a voice clip.
- Voice transcribed correctly to "is my flight delayed."
- Result: confidence 0.00, message correctly explains flight status needs
  real-time airline data and can't be answered by this system -- it did
  NOT fall back to the restroom image and give a wrong/irrelevant answer.
- This is a genuine "modality conflict" test case (image says restroom,
  voice says flight status) and the system handled it well by correctly
  prioritizing the clear out-of-scope voice intent and refusing honestly,
  rather than blending both into a confused or overconfident answer.
- Good evidence for the report's "graceful degradation / honest
  uncertainty" section, now demonstrated in a combined-modality context.

### Suggested test-scenario table entry
- Input modality: Voice + image (combined)
- User input: Restroom sign photo + voice "is my flight delayed"
- Expected response: Refusal/deflection (flight status out of scope)
- Actual response: Confidence 0.00, correct deflection message
- Correctness: Correct -- did not get confused by conflicting modalities
- Observed limitation: None; this is a clean pass

## Voice + image combined, "agreeing" case -- tested, revealed a transcription bug

- Generated `data/audio/audio_006.wav` via macOS `say` TTS with intended
  text "Where is the nearest restroom?" and attached it together with the
  restroom sign photo.
- Whisper mis-transcribed the audio as "Where is the nearest
  **restaurant**?" (restroom -> restaurant) -- likely because the
  synthetic TTS voice pronunciation of "restroom" was ambiguous enough for
  Whisper to confuse it with a phonetically similar, more common word.
- Despite the wrong transcript, the final result was still correct:
  Restroom Block A, confidence 0.55 -- identical to the image-only test.
  This strongly suggests the image signal dominated the fusion decision
  here, since the (wrong) audio keyword "restaurant" did not pull the
  result toward a restaurant-category answer or change the confidence at
  all.
- This is a genuinely useful, honest finding for the report: it shows (a)
  a real limitation of Whisper transcription on synthetic/TTS-generated
  speech, and (b) that the fusion engine appears robust to a single noisy
  modality when another modality (image) gives a clear, correct signal --
  though it also means confidence isn't increasing when modalities
  "agree," since in this case they didn't actually agree at the text
  level.

### Suggested test-scenario table entry
- Input modality: Voice + image (combined)
- User input: Restroom sign photo + voice intended as "where is the
  nearest restroom" (TTS-generated)
- Expected response: Restroom Block A, ideally higher confidence than
  image alone (0.55) if modalities reinforce
- Actual response: Restroom Block A, confidence 0.55 (same as image alone)
- Correctness: Correct final answer, but audio was mis-transcribed as
  "restaurant" -- image alone appears to have driven the result
- Observed limitation: (1) Whisper mis-transcribed synthetic TTS speech
  ("restroom" heard as "restaurant"); (2) confidence fusion doesn't
  appear to boost score even when it should reinforce, at least not
  detectably here since the audio signal was actually wrong
- Suggested improvement: Test with real human speech instead of TTS for
  more reliable transcription; investigate whether the fusion engine
  applies any confidence boost logic when text/audio and image both
  point to the same category

## Still to test

- Text + image combined (retry with text that reinforces or conflicts
  with the image, to test real fusion behavior vs. just "text doesn't hurt")
- Voice + image combined, agreeing case with REAL human speech (not TTS)
  saying "restroom" clearly, to rule out the TTS mis-transcription issue

---

## UPDATE: CLIP is now working

- Root cause confirmed: torch 2.3.0 had an Apple Silicon compatibility bug
  causing a segfault when loading CLIP weights via `from_pretrained`.
- Fix: upgraded torch to >=2.4 (`pip install --upgrade "torch>=2.4"
  torchvision`). After the upgrade, CLIP loaded successfully
  (`CLIPModel.from_pretrained('openai/clip-vit-base-patch32')` -> "CLIP
  model loaded OK").
- Re-ran the full pytest suite after the torch upgrade -- all 11 tests
  still passed, confirming the upgrade did not break numpy/faiss/whisper
  compatibility.
- `USE_CLIP` set back to `True` in `src/config.py`. This is the current,
  final state -- the system now uses real CLIP embeddings
  (512-dimensional) for image retrieval instead of the histogram fallback
  (98-dimensional).

### Image test results (post-CLIP)

| Test | Image type | Confidence | Result |
|---|---|---|---|
| Test 1 | Synthetic dataset icon (already in training data, blue "RESTROOM" icon) | 1.00 | Correct -- expected, since it's an exact/near-exact match to indexed data |
| Test 2 | Real-world photo, completely different style (physical "WOMEN / Wahine" sign with wheelchair icon, real lighting/background, nothing like the synthetic training icons) | 0.55 | Correct category (restroom) retrieved despite zero visual similarity to training data -- genuine semantic generalization, not memorization |

This is strong, legitimate evidence that CLIP is doing real semantic
understanding rather than exact-match memorization, and that the
confidence score is honestly calibrated (dropping on a harder, more
dissimilar real-world example rather than staying falsely at 1.00).

### Suggested test-scenario table entry (for the report)
- Input modality: Image only
- User input: Real photo of physical "WOMEN/Wahine" restroom sign with
  accessibility icon
- Expected response: Restroom-related KB record
- Actual response: Restroom Block A, confidence 0.55
- Correctness: Correct category, correctly non-overconfident
- Observed limitation: Confidence drops notably on stylistically
  dissimilar real-world images vs. synthetic training data
- Suggested improvement: Expand the reference/training image set with
  more real-world photo examples per category to improve confidence
  calibration on real photos

### Original debugging notes (kept for reference / report writing)
- Isolated the exact failure point before the fix: torch, faiss, and
  transformers all imported fine individually and together. The crash
  happened specifically when loading the actual CLIP model weights via
  `from_pretrained`.
- Tried `use_safetensors=True` as an initial workaround -- hit a separate,
  unrelated bug in the installed `huggingface_hub` version when it
  attempted to call the HF API to auto-convert weights
  (`ValueError: dictionary update sequence element #0 has length 1; 2 is
  required`). This path was abandoned in favor of the torch upgrade.

### How to write this up in the report
This is legitimate, valuable content for the "model design" and
"evaluation" sections of the assignment:

> "CLIP (openai/clip-vit-base-patch32) was used as the vision backbone,
> per the assignment's recommended approach. Initial attempts to load the
> model segfaulted on the test machine (Apple Silicon, torch 2.3.0), a
> known class of PyTorch/Transformers compatibility issue; upgrading to
> torch >=2.4 resolved this. Testing showed CLIP correctly retrieves the
> right knowledge-base category even for real-world photographic images
> that are visually very different from the synthetic training icons
> (e.g. a physical restroom sign photo, confidence 0.55), compared to a
> perfect 1.00 confidence on an already-indexed synthetic icon --
> demonstrating genuine semantic generalization rather than simple
> memorization, and honest confidence calibration on harder examples."

---

## Assignment scope reminder

- This chatbot is NOT expected to answer general-purpose/non-airport
  questions well. It is a closed-domain retrieval system by design.
- What IS expected: graceful handling of out-of-scope input -- i.e.
  correctly returning a low-confidence / "uncertain" response instead of
  confidently hallucinating a wrong airport answer. This was already
  confirmed working (see "tell me a joke" test above).
- This matches the rubric's explicit requirements: "Clear error message
  when the system is uncertain" (Deployment section) and "avoid false
  certainty... when flight, gate, or service information may change"
  (Ethical Considerations section).

---

---

## Session: August 23, 2026 (late evening)

### 6. Full pytest suite run — 14/15 passing
- Ran all 15 tests in `tests/test_pipeline.py` after all previous fixes.
- Result: 14/15 passed.
- Failing test: `charging_station` intent — query "Where can I charge my
  phone?" was not retrieving KB023. Root cause: KB023's `short_description`
  used technical language ("USB-A, USB-C and universal power sockets") with
  no colloquial synonyms, and there was only one query variant in the
  training data.

### 7. charging_station coverage fix — now 15/15
- `data/knowledge_base/airport_kb.json` (KB023 `short_description`):
  Added natural-language synonyms: "Charge your phone, laptop or tablet
  here. Power outlets, USB socket, battery top-up, dead phone, low battery."
- `data/text/passenger_queries.json`: Added 4 new colloquial query variants
  for `charging_station` intent (dead phone, power outlet, charge laptop,
  low battery socket).
- Re-ran targeted 6-test suite: 6/6 passed including all new variants.
- Re-ran full suite: 15/15 passed. ✅

### 8. Confidence threshold lowered for offline/TF-IDF mode
- `src/config.py`: `CONFIDENCE_LOW_THRESHOLD` lowered from **0.35 → 0.20**.
- Root cause: with CLIP/sentence-transformers unavailable (offline), the
  TF-IDF + keyword fallback pipeline produces naturally lower similarity
  scores (0.20–0.45 range for correct matches). The old threshold of 0.35
  was causing correct KB matches to show the low-confidence fallback message
  ("please visit the Information Desk") instead of the real answer.
- Confirmed working live in the browser: "where can i charge my phone"
  now returns the correct KB023 answer (Charging Station Zone, Level 2,
  Gate areas B and C) with confidence 0.32 — no longer triggering the
  fallback message.

### Live browser test (post-fix)
| Query | KB | Confidence | Result |
|---|---|---|---|
| "where can i charge my phone" | KB023 | 0.32 | ✅ Correct answer shown |

---

## Suggested next steps (in order)

1. Finish testing remaining scenarios: image only, text+image, voice+image.
2. Decide whether to attempt the torch upgrade to fix CLIP, or keep the
   fallback descriptor and document it as a limitation (recommended if
   time is limited).
3. Build the required 5+ structured test scenarios table for the
   Deployment & User Testing section of the report (input modality, user
   input, expected response, actual response, correctness, limitation,
   suggested improvement).
4. Write up the evaluation section using the real WER data and any
   image/fusion test results collected.
5. Address the Ethical & Regulatory Considerations section, using the
   uncertainty-handling and flight-status-redirect behavior as concrete
   examples of avoiding false certainty.
