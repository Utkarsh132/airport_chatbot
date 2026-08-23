"""Small formatting helpers for the Streamlit UI."""

def confidence_color(confidence: float) -> str:
    if confidence >= 0.65:
        return "green"
    if confidence >= 0.35:
        return "orange"
    return "red"


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
