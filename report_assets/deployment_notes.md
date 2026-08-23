# Deployment and User Testing Notes

## Prototype
Streamlit was chosen over Flask/Gradio for rapid prototyping with built-in file upload widgets,
session-free state management (important since this sandboxed environment disallows browser
storage), and minimal boilerplate for a student demo.

## Structured Test Scenarios
See `report_assets/tables.md` for the 5+ structured test scenarios covering text, voice, image,
and combined inputs, each with expected vs. actual response and observed limitations.

## Verified Behavior (from actual test runs)
- Text-only query "Where is gate B12?" correctly resolved to KB001 (Gate B12) with confidence 1.00.
- Image-only query (synthetic gate sign) correctly classified as category "gate" and resolved to
  the same record with confidence 1.00, confirming the vision fallback pipeline works even without
  CLIP/torch installed.
- Both pathways produced consistent, well-formatted natural language responses including
  directions, hours, and accessibility information.

## Known UI Limitations
- No persistent chat history (by design, to avoid storing passenger queries).
- Uncertainty threshold is a fixed constant rather than adaptively calibrated; a production
  system should calibrate thresholds against a larger labeled validation set.
