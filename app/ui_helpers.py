"""Small formatting helpers for the Streamlit UI."""


def confidence_color(confidence: float) -> str:
    """Return a Streamlit colour name based on confidence band."""
    if confidence >= 0.65:
        return "green"
    if confidence >= 0.20:
        return "orange"
    return "red"


def confidence_bar_html(confidence: float) -> str:
    """Return an HTML progress-bar snippet coloured by confidence band."""
    pct = int(confidence * 100)
    if confidence >= 0.65:
        bar_color = "#4caf50"   # green
        label_color = "#4caf50"
        label = "High confidence"
    elif confidence >= 0.20:
        bar_color = "#ff9800"   # amber
        label_color = "#ff9800"
        label = "Medium confidence"
    else:
        bar_color = "#f44336"   # red
        label_color = "#f44336"
        label = "Low confidence"

    return f"""
<div style="margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="font-size:0.78rem;font-weight:600;color:{label_color};letter-spacing:0.03em;">
      {label}
    </span>
    <span style="font-size:0.78rem;color:#888;">{pct}%</span>
  </div>
  <div style="background:#2a2a2a;border-radius:6px;height:7px;overflow:hidden;">
    <div style="width:{pct}%;background:{bar_color};height:100%;border-radius:6px;
                transition:width 0.4s ease;"></div>
  </div>
</div>
"""


def format_record_card(record: dict) -> str:
    if not record:
        return "_No matching record found._"
    return (
        f"**{record['service_name']}**  \n"
        f"Category: `{record['category']}`  \n"
        f"Location: {record['floor_or_zone']}, {record['terminal']}  \n"
        f"Hours: {record['opening_hours']}  \n"
        f"Directions: {record['directions']}  \n"
        f"Accessibility: {record['accessibility']}  \n"
        f"Related facilities: {record['related_facilities']}  \n"
        f"Assistance contact: {record['emergency_or_help_contact']}"
    )


EXAMPLE_QUERIES = [
    "Where is gate B12?",
    "How do I get to baggage claim?",
    "Is there a lounge near terminal 2?",
    "Where can I find airport transport?",
    "Where is the nearest information desk?",
    "I need special assistance, I use a wheelchair.",
]

PRIVACY_NOTICE = (
    "This demo does not permanently store uploaded images or audio. "
    "Please do not upload boarding passes, passports, or other documents "
    "containing personal data. Responses may be inaccurate -- always verify "
    "critical flight information with official airport displays or staff."
)
