# Ethical and Regulatory Considerations

This discussion is directly tied to the design of the airport multimodal chatbot, not written
as generic AI ethics theory.

## Data Privacy for Voice, Images, and Boarding Pass Data
The chatbot accepts image and audio uploads that could inadvertently contain personal data --
a passenger might photograph their boarding pass (containing name, PNR/booking reference,
flight number) instead of just a sign, or a voice recording might include a companion's name.
**Design response**: the Streamlit UI displays a persistent privacy notice before any input
(`ui_helpers.PRIVACY_NOTICE`) explicitly warning users not to upload boarding passes, passports,
or other documents with personal data. Uploaded files are written to a temporary path only for
the duration of inference and are not persisted to a database.

## GDPR Principles: Data Minimisation, Consent, Secure Handling
- **Data minimisation**: the system only extracts the minimum entities needed to answer a query
  (gate, terminal, flight number, time) via regex-based extraction in `text_pipeline.py` --
  it does not attempt to identify or store passenger identity.
- **Consent**: the UI-level privacy notice functions as an explicit, visible consent/warning
  mechanism before any upload, rather than a buried terms-of-service link.
- **Secure handling**: no long-term storage is implemented by default; the fusion engine
  (`fusion.py`) processes inputs in-memory per request. A production deployment would need
  encrypted transient storage, EU-region hosting, and automatic deletion of uploaded files
  immediately after inference.

## Risks of Processing Sensitive Passenger Details
Names, flight numbers, and booking references are highly identifying when combined. The system
deliberately avoids querying or returning any passenger-specific data -- the knowledge base
(`airport_kb.json`) contains only static facility information, never passenger records. Flight
status queries ("Is my flight delayed?") are intentionally routed to an uncertainty response
rather than fabricated, since answering would require live PNR lookups against a real airline
system that is out of scope and carries elevated privacy risk.

## Bias in Speech Recognition Across Accents, Languages, Noisy Environments
The speech evaluation (`evaluation.py::evaluate_speech`) explicitly documents accent and noise as
error sources (`audio_pipeline.COMMON_TRANSCRIPTION_ERROR_SOURCES`). Whisper's accuracy is known
to vary across accents and background noise conditions typical of a busy terminal (PA
announcements, crowd noise). **Mitigation**: the fusion engine treats a low-confidence or empty
transcript as insufficient evidence and falls back to the uncertainty message rather than guessing,
and the UI displays the raw transcript back to the user so they can verify or correct it via text.

## Accessibility for Passengers with Disabilities
Every knowledge base record includes an explicit `accessibility` field (step-free access,
wheelchair ramps, hearing loops, accessible counters), and one dedicated record (Special
Assistance Desk) exists specifically for passengers needing mobility, visual, hearing, or
cognitive support. The multimodal design itself is an accessibility feature: passengers who
cannot easily type can use voice, and passengers unfamiliar with the airport's language can point
a camera at a sign instead.

## Limitations of Multilingual Support
The current proof-of-concept's rule-based intent classifier and regex entity extractor are
English-only; a non-English query would likely be misclassified as "unknown" intent. This is a
documented limitation. **Recommended improvement**: replace keyword-based intent matching with a
multilingual sentence-transformer model and route Whisper's language detection output to select
appropriate response templates.

## Avoiding False Certainty When Information May Change
Gate assignments, security wait times, and flight status change in real time, but the knowledge
base is a static snapshot. The fusion engine enforces a confidence threshold
(`CONFIDENCE_LOW_THRESHOLD`) below which it explicitly states uncertainty and redirects the user,
and even confident static-location answers (e.g., "Gate B12 directions") are distinct from
volatile data (e.g., actual departure time), which the system never fabricates.

## Human Handover and Redirection to Official Sources
Every uncertain response and every flight-status-type query explicitly instructs the user to
"visit the nearest Information Desk or check the official airport app/display screens," ensuring
the chatbot never presents itself as a sole authoritative source for safety- or time-critical
information.

## Concrete Design Actions Implemented
- No long-term storage of uploaded images/audio by default (in-memory processing only).
- Local/offline-capable processing via lightweight fallback models, reducing third-party data
  exposure risk.
- Explicit in-UI warning against uploading sensitive documents.
- Confidence thresholding with a hard-coded low-confidence cutoff triggering human fallback.
- Human handover recommendation embedded directly in uncertain and flight-status responses.
