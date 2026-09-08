"""
simulate_logs.py
================
Generates a realistic multi-user session log for the Airport Multimodal Chatbot.

Grounded in the REAL knowledge base (data/knowledge_base/airport_kb.json) and
the REAL passenger query dataset (data/text/passenger_queries.json) of the
project so that all downstream analytics (intent accuracy, fallback rate,
co-occurrence, LTV, funnel, frustration) reflect the actual bot design.

Output:
    outputs/logs/conversation_logs.csv     -- one row per user turn
    outputs/logs/sessions.csv              -- one row per session
    outputs/logs/users.csv                 -- one row per user

Design choices (documented in report):
  * 420 sessions across 260 unique users (some repeat visitors -> LTV analysis)
  * Traveller personas: business, leisure, family, transit, first-time
  * Channels: web, mobile, kiosk, in-app
  * Modalities: text, voice, image (matches bot design)
  * Ground truth intent is stored, then predicted intent is drawn from a
    confusion distribution tuned per intent difficulty (rare/OOS harder)
  * Fallback fires when confidence < FALLBACK_THRESHOLD (0.45)
  * Response times drawn from a modality-aware log-normal
  * CSAT (1-5) drawn from a model conditioned on: correct intent, fallback,
    response time, and frustration signals in the utterance
"""

from __future__ import annotations
import json, csv, random, uuid, math
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTDIR = ROOT / "outputs" / "logs"
OUTDIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ---------------------------------------------------------------------------
# Load real KB + intent seeds
# ---------------------------------------------------------------------------
kb = json.load(open(INPUTS / "airport_kb.json"))
queries = json.load(open(INPUTS / "passenger_queries.json"))

# map intent -> list of example utterances
INTENT_TO_TEXTS: dict[str, list[str]] = {}
for q in queries:
    INTENT_TO_TEXTS.setdefault(q["intent"], []).append(q["text"])

ALL_INTENTS = sorted(INTENT_TO_TEXTS.keys())
IN_SCOPE_INTENTS = [i for i in ALL_INTENTS if i != "out_of_scope"]

# frustration templates layered on top of a normal query
FRUSTRATION_PHRASES = [
    "this is useless", "you're not helping", "I already asked this",
    "just tell me", "come on", "seriously?", "for the third time,",
    "why is this so hard", "ugh", "hello?? are you there",
]
REPEAT_MARKERS = ["again", "still", "as I said", "one more time"]

# Personas drive intent mix
PERSONAS = {
    "business":    dict(share=0.22, sessions_mu=2.3,
        top=["find_gate","lounge","wifi","boarding_time","fast_track" if False else "security",
             "flight_status","charging_station","transfer_desk","transit_connection"]),
    "leisure":     dict(share=0.34, sessions_mu=1.4,
        top=["find_gate","restaurant","shopping","duty_free","restroom",
             "check_in","boarding_time","currency_exchange","information_desk"]),
    "family":      dict(share=0.18, sessions_mu=1.5,
        top=["kids_play_area","family_room","restroom","restaurant","find_gate",
             "baggage_claim","lost_child","information_desk","stroller" if False else "special_assistance"]),
    "transit":     dict(share=0.14, sessions_mu=1.7,
        top=["transit_connection","transfer_desk","find_gate","lounge","shower_facility",
             "customs","immigration","charging_station"]),
    "first_time":  dict(share=0.12, sessions_mu=1.2,
        top=["check_in","security","find_gate","information_desk","baggage_claim",
             "restroom","transport","currency_exchange","customs"]),
}
# Normalise persona shares
_s = sum(p["share"] for p in PERSONAS.values())
for p in PERSONAS.values():
    p["share"] /= _s

# Clean each persona's `top` list to intents that actually exist
for pname, p in PERSONAS.items():
    p["top"] = [i for i in p["top"] if i in IN_SCOPE_INTENTS]

CHANNEL_DIST = {"mobile_app": 0.48, "web": 0.28, "kiosk": 0.14, "in_terminal_wifi": 0.10}
MODALITY_DIST_BY_CHANNEL = {
    "mobile_app":       {"text": 0.55, "voice": 0.28, "image": 0.17},
    "web":              {"text": 0.82, "voice": 0.06, "image": 0.12},
    "kiosk":            {"text": 0.72, "voice": 0.04, "image": 0.24},
    "in_terminal_wifi": {"text": 0.66, "voice": 0.22, "image": 0.12},
}

# Intent difficulty (higher = more confusable / more likely to be misclassified)
# Manually tuned: rare, ambiguous, or wordy intents are harder.
HARD_INTENTS = {"transit_connection","transfer_desk","special_assistance","gate_change",
                "hotel_airside","hotel_shuttle","visa_on_arrival","immigration",
                "baggage_wrapping","oversized_baggage","emergency_exit","lost_child",
                "medical_assistance","police_desk","unattended_baggage","halal_food",
                "self_checkin_kiosk","luggage_trolley","water_fountain","vending_machine"}

# Confusion map: hard intents most often mis-classified as these look-alikes
CONFUSION_MAP = {
    "transit_connection": ["transfer_desk","find_gate","boarding_time"],
    "transfer_desk":      ["transit_connection","information_desk","check_in"],
    "gate_change":        ["find_gate","boarding_time","flight_status"],
    "boarding_time":      ["flight_status","find_gate"],
    "hotel_airside":      ["lounge","sleeping_area"],
    "hotel_shuttle":      ["transport","information_desk"],
    "visa_on_arrival":    ["customs","immigration","information_desk"],
    "immigration":        ["customs","visa_on_arrival"],
    "customs":            ["immigration","baggage_claim"],
    "baggage_wrapping":   ["baggage_drop","baggage_claim","baggage_services" if False else "check_in"],
    "oversized_baggage":  ["baggage_drop","check_in"],
    "lost_child":         ["lost_and_found","information_desk","special_assistance"],
    "lost_and_found":     ["lost_child","information_desk"],
    "medical_assistance": ["information_desk","special_assistance","pharmacy"],
    "police_desk":        ["information_desk","emergency_exit"],
    "special_assistance": ["information_desk","medical_assistance"],
    "halal_food":         ["restaurant","coffee_shop"],
    "self_checkin_kiosk": ["check_in","information_desk"],
    "luggage_trolley":    ["baggage_claim","baggage_drop"],
    "water_fountain":     ["restroom","restaurant"],
    "vending_machine":    ["coffee_shop","restaurant","shopping"],
    "emergency_exit":     ["information_desk","police_desk"],
    "unattended_baggage": ["police_desk","information_desk"],
    "prayer_room":        ["lounge","sleeping_area"],
    "shower_facility":    ["lounge","restroom"],
    "fast_track" if False else "atm":  ["currency_exchange","information_desk"],
    "coffee_shop":        ["restaurant","duty_free"],
    "duty_free":          ["shopping","coffee_shop"],
    "wifi":               ["information_desk","charging_station"],
    "charging_station":   ["wifi","information_desk"],
    "uber_pickup":        ["transport","bus_stop"],
    "bus_stop":           ["transport","uber_pickup"],
    "car_rental":         ["transport","information_desk"],
    "bag_storage":        ["baggage_claim","information_desk"],
    "flight_status":      ["boarding_time","find_gate"],
    "find_gate":          ["boarding_time","gate_change"],
    "check_in":           ["baggage_drop","self_checkin_kiosk"],
    "baggage_drop":       ["check_in","baggage_claim"],
    "baggage_claim":      ["lost_and_found","customs"],
    "security":           ["check_in","information_desk"],
    "lounge":             ["sleeping_area","shower_facility"],
    "sleeping_area":      ["lounge","shower_facility"],
    "restaurant":         ["coffee_shop","duty_free"],
    "restroom":           ["family_room","water_fountain"],
    "family_room":        ["restroom","kids_play_area"],
    "kids_play_area":     ["family_room","restaurant"],
    "shopping":           ["duty_free","currency_exchange"],
    "currency_exchange":  ["atm","information_desk"],
    "parking":            ["transport","car_rental"],
    "transport":          ["parking","uber_pickup"],
    "information_desk":   ["find_gate","flight_status"],
    "pharmacy":           ["medical_assistance","information_desk"],
    "smoking_area":       ["information_desk","lounge"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def weighted_choice(d: dict) -> str:
    keys = list(d.keys())
    weights = [d[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]

def sample_persona() -> str:
    return random.choices(list(PERSONAS.keys()),
                          weights=[p["share"] for p in PERSONAS.values()], k=1)[0]

def sample_intent(persona: str, prev_intent: str | None) -> str:
    """Draw an intent given persona and the immediately preceding intent.

    Persona weights drive the base distribution; the previous intent boosts a
    small co-occurrence set (creates realistic patterns for the Apriori step).
    """
    p = PERSONAS[persona]
    weights = {}
    for i in IN_SCOPE_INTENTS:
        weights[i] = 1.0
    for i in p["top"]:
        weights[i] = weights.get(i, 1.0) + 6.0
    # tiny prob of out-of-scope
    weights["out_of_scope"] = 0.6
    # co-occurrence boosts: common airport journeys
    JOURNEY_LINKS = {
        "check_in":       ["baggage_drop","security","find_gate"],
        "baggage_drop":   ["security","find_gate","lounge"],
        "security":       ["find_gate","duty_free","lounge","restroom"],
        "find_gate":      ["boarding_time","restroom","coffee_shop","duty_free"],
        "boarding_time":  ["find_gate","gate_change"],
        "immigration":    ["customs","baggage_claim"],
        "customs":        ["baggage_claim","lost_and_found"],
        "baggage_claim":  ["lost_and_found","transport","uber_pickup","bus_stop"],
        "lounge":         ["shower_facility","wifi","charging_station","restaurant"],
        "restaurant":     ["duty_free","coffee_shop","find_gate"],
        "shopping":       ["duty_free","currency_exchange"],
        "kids_play_area": ["family_room","restaurant","restroom"],
    }
    if prev_intent and prev_intent in JOURNEY_LINKS:
        for j in JOURNEY_LINKS[prev_intent]:
            if j in weights:
                weights[j] += 5.0
    return weighted_choice(weights)

def sample_modality(channel: str) -> str:
    return weighted_choice(MODALITY_DIST_BY_CHANNEL[channel])

def sample_confidence(intent: str, modality: str) -> float:
    base = 0.82 if intent not in HARD_INTENTS else 0.55
    if modality == "voice":
        base -= 0.06  # ASR error drag
    if modality == "image":
        base -= 0.04  # CLIP retrieval drag
    noise = random.gauss(0, 0.12)
    return max(0.05, min(0.99, base + noise))

def sample_predicted_intent(true_intent: str, confidence: float) -> str:
    """Model the classifier: high conf -> correct; low conf -> pick from confusion map."""
    if confidence >= 0.55 and random.random() < 0.92:
        return true_intent
    # otherwise the classifier tends to slip to a look-alike
    alts = CONFUSION_MAP.get(true_intent, ["information_desk"])
    alts = [a for a in alts if a in ALL_INTENTS] or ["information_desk"]
    if random.random() < 0.15:
        # occasional totally wrong guess
        return random.choice(ALL_INTENTS)
    return random.choice(alts)

def sample_response_time_ms(modality: str, fallback: bool) -> int:
    mu = {"text": 6.6, "voice": 7.4, "image": 7.2}[modality]  # ln(ms)
    sigma = 0.35
    val = math.exp(random.gauss(mu, sigma))
    if fallback:
        val *= 1.4
    return int(val)

def maybe_add_frustration(text: str, prob: float) -> tuple[str, bool]:
    if random.random() < prob:
        pieces = []
        if random.random() < 0.5:
            pieces.append(random.choice(REPEAT_MARKERS))
        pieces.append(text)
        if random.random() < 0.7:
            pieces.append(random.choice(FRUSTRATION_PHRASES))
        return " ".join(pieces).strip(), True
    return text, False

def csat_score(correct: bool, fallback: bool, rt_ms: int, frustrated: bool) -> int:
    """Simulate a 1-5 CSAT rating conditioned on turn quality."""
    base = 4.6 if correct else 2.3
    if fallback:  base -= 0.9
    if frustrated: base -= 0.7
    if rt_ms > 4000: base -= 0.4
    if rt_ms > 8000: base -= 0.4
    base += random.gauss(0, 0.35)
    return max(1, min(5, int(round(base))))

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
FALLBACK_THRESHOLD = 0.45
N_USERS = 260
N_SESSIONS_TARGET = 420
START_DATE = datetime(2026, 8, 25, 5, 0, 0)  # ~2 weeks of traffic ending near issue

users = []
for _ in range(N_USERS):
    persona = sample_persona()
    users.append({
        "user_id": f"U{uuid.uuid4().hex[:8]}",
        "persona": persona,
        "signup_channel": weighted_choice(CHANNEL_DIST),
    })

# Assign sessions per user (Poisson-ish; capped)
sessions = []
turns = []
sess_left = N_SESSIONS_TARGET
for u in users:
    if sess_left <= 0: break
    mu = PERSONAS[u["persona"]]["sessions_mu"]
    n_sess = max(1, min(6, int(round(random.expovariate(1/mu)))))
    for _ in range(n_sess):
        if sess_left <= 0: break
        sess_left -= 1
        channel = weighted_choice(CHANNEL_DIST) if random.random() < 0.7 else u["signup_channel"]
        started_at = START_DATE + timedelta(
            days=random.randint(0, 13),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        session_id = f"S{uuid.uuid4().hex[:10]}"
        n_turns = max(1, min(9, int(round(random.gauss(3.4, 1.7)))))
        completed = False
        prev_intent = None
        cursor = started_at
        session_frustration_prob = 0.05
        session_turns = []
        for t_idx in range(n_turns):
            intent = sample_intent(u["persona"], prev_intent)
            modality = sample_modality(channel)
            confidence = sample_confidence(intent, modality)

            # a turn is "fallback" if confidence below threshold OR intent is out_of_scope
            fallback = (confidence < FALLBACK_THRESHOLD) or (intent == "out_of_scope")

            # Determine predicted intent
            pred_intent = "fallback" if fallback else sample_predicted_intent(intent, confidence)
            correct = (pred_intent == intent) and not fallback

            # Utterance text (grounded in real examples)
            base_utt = random.choice(INTENT_TO_TEXTS.get(intent, ["can you help me"]))
            utt, frustrated = maybe_add_frustration(base_utt, session_frustration_prob)

            rt = sample_response_time_ms(modality, fallback)
            csat = csat_score(correct, fallback, rt, frustrated)
            ttfr_ms = int(rt * random.uniform(0.35, 0.6))  # bot's time-to-first-token

            # Escalate frustration prob inside the session when things go wrong
            if fallback or not correct or frustrated:
                session_frustration_prob = min(0.55, session_frustration_prob + 0.13)
            else:
                session_frustration_prob = max(0.03, session_frustration_prob - 0.03)

            # completion signal: last turn of a happy path
            if t_idx == n_turns - 1 and correct and not fallback and csat >= 4:
                completed = True

            session_turns.append({
                "user_id": u["user_id"],
                "session_id": session_id,
                "turn_index": t_idx,
                "timestamp": cursor.isoformat(timespec="seconds"),
                "channel": channel,
                "modality": modality,
                "persona": u["persona"],
                "utterance": utt,
                "true_intent": intent,
                "predicted_intent": pred_intent,
                "confidence": round(confidence, 3),
                "is_fallback": fallback,
                "is_correct": correct,
                "response_time_ms": rt,
                "time_to_first_response_ms": ttfr_ms,
                "csat": csat,
                "frustrated_signal": frustrated,
            })
            prev_intent = intent
            cursor = cursor + timedelta(seconds=random.randint(8, 90))

        # Some users abandon: mark abandoned if last turn was fallback / bad
        last = session_turns[-1]
        abandoned = last["is_fallback"] or (last["csat"] <= 2 and not completed)

        turns.extend(session_turns)
        sessions.append({
            "session_id": session_id,
            "user_id": u["user_id"],
            "persona": u["persona"],
            "channel": channel,
            "started_at": started_at.isoformat(timespec="seconds"),
            "n_turns": n_turns,
            "completed": completed,
            "abandoned": abandoned,
            "avg_csat": round(sum(t["csat"] for t in session_turns)/len(session_turns), 2),
            "any_fallback": any(t["is_fallback"] for t in session_turns),
            "any_frustration": any(t["frustrated_signal"] for t in session_turns),
        })

# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------
def write_csv(path, rows, fieldnames=None):
    if not rows:
        path.write_text("")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

write_csv(OUTDIR / "users.csv", users)
write_csv(OUTDIR / "sessions.csv", sessions)
write_csv(OUTDIR / "conversation_logs.csv", turns)

print(f"Generated {len(users)} users, {len(sessions)} sessions, {len(turns)} turns")
print(f"Saved to {OUTDIR}/")
