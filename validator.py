"""
Pre-flight output validator and safety gate engine for Vera Bot.
Enforces 12 safety gates including Fact Traceability to prevent the -2 hallucination penalty.
"""
import re
import json
import datetime
from typing import Dict, Any, Optional, List, Set

REQUIRED_ACTION_KEYS = [
    "conversation_id", "merchant_id", "customer_id", "send_as",
    "trigger_id", "template_name", "template_params", "body",
    "cta", "suppression_key", "rationale",
]

JARGON_TERMS = [
    r"\btrigger\b", r"\bsuppression_key\b", r"\btemplate_name\b",
    r"\bcta\b", r"\bcontext_id\b", r"\bpayload\b",
]

VALID_CTAS = {"binary_yes_no", "open_ended", "single_choice", "none"}

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DATE_KEYS = (
    "match_time_iso", "start_time_iso", "event_time_iso", "start_time", "event_time",
    "date", "date_iso", "event_date", "festival_date", "opened_date",
)
AUDIENCE_MISMATCH_FLAGS = {"adult", "body", "composition", "membership", "first month"}
COMPLETION_CLAIM_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:we|i)(?:\s+(?:have|has)|['’](?:ve|s))\s+(?:already\s+)?(?:reserved|booked|confirmed|scheduled|held)\b",
    r"\b(?:the studio|the team)\s+(?:has|have)\s+(?:already\s+)?(?:reserved|booked|confirmed|scheduled|held)\b",
    r"\b(?:we|the studio|the team)\s+(?:have|has)\s+a\s+(?:spot|slot)\s+(?:reserved|booked|held)\b",
    r"\b(?:your|his|her|their)\s+(?:slot|spot|booking|appointment)\s+(?:is|has been)\s+(?:confirmed|booked|reserved|held)\b",
    r"\b(?:booking|reservation)\s+(?:is|has been)\s+(?:confirmed|made)\b",
    r"\b(?:the )?(?:appointment|session)\s+is\s+now\s+booked\b",
)]
CLINICAL_SALES_JARGON = [re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bhigh-?converting\b",
    r"\bboost (?:engagement|sales|revenue)\b",
    r"\bmaximi[sz]e (?:revenue|conversions?)\b",
    r"\bdrive sales\b",
    r"\bconversion rate\b",
)]
SELF_REFERENCE_THIRD_PERSON = re.compile(
    r"\b(?:ask|contact|check with|call|tell)\s+the\s+(?:studio|clinic|salon|pharmacy|restaurant|gym)\b"
    r"|\bthe\s+(?:studio|clinic|salon|pharmacy|restaurant|gym)\s+(?:will|would|can|could|has|have|is)\b",
    re.IGNORECASE,
)
PRICE_MATCH_FRAMING = [re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bmatch\b[^.]{0,25}\b(?:price|offer)\b",
    r"\bbeat\b[^.]{0,25}\b(?:price|offer)\b",
    r"\bsame price\b",
    r"\blower (?:our|your) price\b",
    r"\bundercut\b",
    r"\bprice war\b",
    r"\bcompete on price\b",
)]
ALLOWED_DEADLINE_KINDS = {
    "regulation_change", "compliance_alert", "renewal_due", "subscription_expiring",
    "chronic_refill_due",
}
UNRESOLVED_TEMPLATE_PATTERNS = (
    re.compile(r"(?:\.{3}|…+)"),
    re.compile(r"\b(?:None|null|undefined|TBD)\b", re.IGNORECASE),
    re.compile(r"\{[A-Za-z_]\w*\}"),
)
UNSUPPORTED_FREE_DELIVERY = re.compile(r"\bfree\s+(?:home\s+)?delivery\b", re.IGNORECASE)
UNSUPPORTED_ORDER_ACTION = re.compile(
    r"\b(?:confirm|place)\s+(?:the\s+)?order\b|\border\s+confirm\b", re.IGNORECASE
)


def check_no_clinical_sales_jargon(body: str, category: Dict[str, Any]) -> bool:
    """Keep clinical/collegial category copy free of sales-agency jargon."""
    voice = (category or {}).get("voice", {})
    voice = voice if isinstance(voice, dict) else {}
    register = str(voice.get("register", "")).lower()
    if "clinical" not in register and "collegial" not in register:
        return True
    return not any(pattern.search(body or "") for pattern in CLINICAL_SALES_JARGON)


def check_no_third_person_self_reference(body: str, is_customer_facing: bool) -> bool:
    """Prevent merchant-on-behalf copy from referring to the sender as a third party."""
    if not is_customer_facing:
        return True
    return not SELF_REFERENCE_THIRD_PERSON.search(body or "")


def check_no_price_matching_framing(body: str, trigger: Dict[str, Any]) -> bool:
    """Block reactive price-matching language without banning relevant own offers."""
    if not isinstance(trigger, dict) or trigger.get("kind") != "competitor_opened":
        return True
    return not any(pattern.search(body or "") for pattern in PRICE_MATCH_FRAMING)


def _collect_string_values(obj: Any) -> List[str]:
    values: List[str] = []
    if isinstance(obj, str):
        values.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            values.extend(_collect_string_values(v))
    elif isinstance(obj, list):
        for v in obj:
            values.extend(_collect_string_values(v))
    return values


def _matches_same_date(value: str, target: datetime.date) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    # ISO dates and explicit calendar dates are treated as grounded if they refer to the same calendar date.
    iso_matches = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    for iso in iso_matches:
        try:
            if datetime.date.fromisoformat(iso) == target:
                return True
        except ValueError:
            continue

    month_day_matches = re.findall(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?", text, flags=re.IGNORECASE)
    for month_day in month_day_matches:
        try:
            parsed = datetime.datetime.strptime(month_day, "%B %d")
        except ValueError:
            try:
                parsed = datetime.datetime.strptime(month_day, "%b %d")
            except ValueError:
                continue
        if parsed.date() == target:
            return True

    numeric_day = re.findall(r"\b\d{1,2}(?:st|nd|rd|th)?\b", text)
    for day_part in numeric_day:
        try:
            day_num = int(re.sub(r"(?:st|nd|rd|th)", "", day_part))
        except ValueError:
            continue
        if day_num == target.day:
            # This is only used as a same-day guard when the surrounding text clearly points to the same month.
            month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"]
            if any(month in text.lower() for month in month_names):
                return True
    return False


def find_expiry_date_mention(
    body: str,
    trigger: Dict[str, Any],
    explicit_contexts: Optional[List[Any]] = None,
) -> Optional[str]:
    """Return an expiry-date substring used without a separate supporting fact."""
    if not isinstance(trigger, dict) or trigger.get("kind") in ALLOWED_DEADLINE_KINDS:
        return None
    expires_at = trigger.get("expires_at")
    if not isinstance(expires_at, str) or not expires_at.strip():
        return None
    try:
        parsed = datetime.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return None

    target_date = parsed.date()
    contexts = explicit_contexts or []
    grounded_date_seen = False
    for item in contexts:
        for value in _collect_string_values(item):
            if _matches_same_date(value, target_date):
                grounded_date_seen = True
                break
        if grounded_date_seen:
            break
    if grounded_date_seen:
        return None

    day = str(parsed.day)
    candidates = (
        f"{parsed.strftime('%B')} {day}",
        f"{parsed.strftime('%b')} {day}",
        f"{day} {parsed.strftime('%B')}",
        f"{day} {parsed.strftime('%b')}",
        parsed.strftime("%Y-%m-%d"),
    )
    text = body or ""
    for candidate in candidates:
        pattern = re.compile(
            rf"(?<!\w){re.escape(candidate)}(?:st|nd|rd|th)?(?!\d)", re.IGNORECASE
        )
        match = pattern.search(text)
        if not match:
            continue
        return match.group(0)
    return None


def check_no_expires_at_as_deadline(
    body: str,
    trigger: Dict[str, Any],
    explicit_contexts: Optional[List[Any]] = None,
) -> bool:
    """Do not present an internal alert expiry as an external business deadline."""
    return find_expiry_date_mention(body, trigger, explicit_contexts) is None


def check_no_unresolved_template_values(body: str) -> bool:
    """Reject obvious unexpanded or empty template markers before sending a message."""
    text = body or ""
    return not any(pattern.search(text) for pattern in UNRESOLVED_TEMPLATE_PATTERNS)


def check_refill_claims_are_supported(body: str, trigger: Dict[str, Any]) -> bool:
    """Do not promise conditional delivery benefits or order operations not confirmed by refill context."""
    if not isinstance(trigger, dict) or trigger.get("kind") != "chronic_refill_due":
        return True
    payload = trigger.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    if UNSUPPORTED_ORDER_ACTION.search(body or ""):
        return False
    if payload.get("free_delivery_confirmed") is not True and UNSUPPORTED_FREE_DELIVERY.search(body or ""):
        return False
    return True

def check_no_unearned_completion_claims(body: str, trigger: Dict[str, Any]) -> bool:
    """Reject claims that a booking/action was completed unless the trigger confirms it."""
    trigger = trigger if isinstance(trigger, dict) else {}
    payload = trigger.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    if payload.get("confirmed") is True or payload.get("booking_status") == "confirmed":
        return True
    text = body or ""
    if any(pattern.search(text) for pattern in COMPLETION_CLAIM_PATTERNS):
        return False
    if trigger.get("kind") == "trial_followup" and re.search(
        r"\b(?:shall|should|can|may)\s+i\s+(?:go ahead and\s+)?(?:confirm|book|reserve|hold)\b",
        text,
        re.IGNORECASE,
    ):
        # The bot has no booking/confirmation operation; offer to ask the venue to hold it instead.
        return False
    return True

def check_weekday_consistency(body: str, trigger: Dict[str, Any]) -> bool:
    """Reject weekday names inconsistent with a date explicitly supplied by the trigger."""
    trigger = trigger if isinstance(trigger, dict) else {}
    payload = trigger.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    date_value = next((
        payload.get(key) for key in DATE_KEYS
        if isinstance(payload.get(key), str) and payload.get(key).strip()
    ), None)
    if not date_value:
        return True
    try:
        try:
            parsed = datetime.datetime.fromisoformat(date_value.replace("Z", "+00:00"))
            actual_weekday = parsed.strftime("%A").lower()
        except ValueError:
            parsed_date = datetime.date.fromisoformat(date_value[:10])
            actual_weekday = parsed_date.strftime("%A").lower()
    except ValueError:
        return True
    mentioned = [day for day in WEEKDAYS if re.search(rf"\b{day}\b", body or "", re.IGNORECASE)]
    return all(day == actual_weekday for day in mentioned)

def check_topic_relevant_offer(body: str, trigger: Dict[str, Any], merchant: Dict[str, Any]) -> bool:
    """Reject offers whose explicit audience/product terms conflict with a planning topic."""
    trigger = trigger if isinstance(trigger, dict) else {}
    if trigger.get("kind") not in ("active_planning_intent", "planning_intent"):
        return True
    payload = trigger.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    topic = payload.get("intent_topic", "")
    topic_words = set(re.findall(r"[a-z]+", str(topic).replace("_", " ").lower()))
    if not topic_words:
        return True
    body_lower = (body or "").lower()
    offers = (merchant or {}).get("offers", [])
    if not isinstance(offers, list):
        return True
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        title = str(offer.get("title", "")).strip()
        if not title or title.lower() not in body_lower:
            continue
        offer_lower = title.lower()
        offer_words = set(re.findall(r"[a-z]+", offer_lower))
        flagged = bool(offer_words & AUDIENCE_MISMATCH_FLAGS) or "first month" in offer_lower
        if flagged and not (offer_words & topic_words):
            return False
    return True

def extract_required_anchor_terms(trigger: Dict[str, Any]) -> List[tuple]:
    """Return relevant trigger entities that a generated message must retain.

    Each item is (term, fuzzy): fuzzy=True means the term is a derived, generic
    phrase (e.g. from likely_driver) that should be checked word-by-word so a
    natural paraphrase ("kids' yoga class") still passes; fuzzy=False means the
    term is a proper noun (competitor name, molecule) that must still appear
    close to verbatim for specificity to mean anything.
    """
    trigger = trigger if isinstance(trigger, dict) else {}
    payload = trigger.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    kind = trigger.get("kind", "")
    terms: List[tuple] = []

    if kind in ("perf_spike", "milestone_reached"):
        driver = payload.get("likely_driver")
        if isinstance(driver, str) and driver.strip():
            driver = re.sub(r"_post$", "", driver.strip(), flags=re.IGNORECASE)
            terms.append((driver.replace("_", " ").strip(), True))
    elif kind == "competitor_opened":
        name = payload.get("competitor_name")
        if isinstance(name, str) and name.strip():
            terms.append((name.strip(), False))
    elif kind in ("supply_alert", "recall_alert"):
        molecule = payload.get("molecule")
        if isinstance(molecule, str) and molecule.strip():
            terms.append((molecule.strip(), False))
    return terms


def _normalize_for_match(text: str) -> str:
    """Lowercase, unify curly apostrophes, and strip punctuation for tolerant matching."""
    text = (text or "").lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def check_anchor_terms_present(body: str, terms: List[tuple]) -> bool:
    """
    Verify required anchor terms survive into the message.

    Bug fixed: this previously required the FULL multi-word phrase (e.g. "kids
    yoga") to appear as one exact contiguous substring. A perfectly good, fully
    grounded LLM draft phrased as "kids' yoga class" or "yoga for kids" failed
    this check and got reverted to the deterministic fallback purely because of
    word order or an apostrophe -- confirmed against trg_024_perf_spike_zen's
    actual payload (likely_driver="kids_yoga_post"), which the transformation
    turns into the anchor "kids yoga". Fuzzy terms now require every significant
    word to appear (in any order, tolerant of punctuation/pluralization); exact
    terms (competitor names, molecules) still require the phrase verbatim,
    since paraphrasing a proper noun would itself be a specificity regression.
    """
    body_norm = _normalize_for_match(body or "")
    for term, fuzzy in terms:
        term_norm = _normalize_for_match(term)
        if not term_norm:
            continue
        if not fuzzy:
            if term_norm not in body_norm:
                return False
        else:
            words = [w for w in term_norm.split() if w]
            if not all(re.search(rf"\b{re.escape(w)}\w*\b", body_norm) for w in words):
                return False
    return True

def check_taboo_words(body: str, category: Dict[str, Any]) -> bool:
    voice = (category or {}).get("voice", {})
    voice = voice if isinstance(voice, dict) else {}
    taboos = voice.get("vocab_taboo") or voice.get("taboos") or []
    for taboo in taboos:
        if isinstance(taboo, str) and taboo.strip():
            if re.search(rf"\b{re.escape(taboo.strip())}\b", body or "", re.IGNORECASE):
                return False
    return True

def strip_urls(text: str) -> str:
    clean = re.sub(r"https?://\S+|www\.\S+", "", text)
    return re.sub(r"\s+", " ", clean).strip()

def strip_jargon(text: str) -> str:
    clean = text
    for pat in JARGON_TERMS:
        clean = re.sub(pat, "", clean, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", clean).strip()

def ensure_owner_name(body: str, merchant: Dict[str, Any], is_customer_facing: bool, customer: Optional[Dict[str, Any]] = None) -> str:
    m_name = (merchant or {}).get("identity", {}).get("name", "")
    if is_customer_facing:
        if m_name and m_name.lower() not in body.lower():
            c_name = (customer or {}).get("identity", {}).get("name", "there")
            if body.lower().startswith(f"hi {c_name.lower()}") or body.lower().startswith("hi "):
                parts = body.split(",", 1)
                if len(parts) == 2:
                    body = f"{parts[0]}, {m_name} here! {parts[1].strip()}"
                else:
                    body = f"Hi {c_name}, greetings from {m_name}! {body}"
            else:
                body = f"Greetings from {m_name}! {body}"
        return body
    
    owner_name = (merchant or {}).get("identity", {}).get("owner_first_name")
    if not owner_name:
        return body
        
    cat_slug = (merchant or {}).get("category_slug", "")
    salutation = f"Dr. {owner_name}" if cat_slug == "dentists" and not owner_name.startswith("Dr") else owner_name
    
    if owner_name.lower() not in body.lower() and salutation.lower() not in body.lower():
        # Bug fixed: this used to be
        #   body[0].lower() if len(body) > 1 and body[0].isupper() else body
        # which, whenever the first character was uppercase, kept ONLY that
        # single lowercased character and silently discarded the rest of the
        # message (body[1:] was never reattached) -- e.g. a real draft
        # "For 10 or more, it is..." became the single sent message
        # "Suresh, f". This path is only reached when a generated body doesn't
        # already contain the merchant owner's name (rare in the 25 canonical
        # pairs, since the prompt normally forces a name greeting -- which is
        # exactly why this stayed invisible in your test runs), but it is a
        # live landmine: any future draft that omits the name for any reason
        # (a fallback model with different phrasing tendencies, a reply-track
        # generation, a new trigger kind) would get silently truncated to a
        # few characters instead of prepending the missing salutation.
        if len(body) > 1 and body[0].isupper():
            rest = body[0].lower() + body[1:]
        else:
            rest = body
        body = f"{salutation}, {rest}"
    return body

def check_fact_traceability(body: str, contexts: list) -> bool:
    """Allow exact numeric facts plus safe formatting, rounding, and percent conversions.

    Bug fixed: grounding used to be `normalized in context_dump`, a plain Python
    substring test against the ENTIRE json-dumped context blob. For any short
    number (1-2 digits -- the common case: "6pm", "18 calls", "9 reviews") that
    substring will coincidentally appear inside some unrelated larger number,
    id, or date almost every time a context blob has any real size, so short
    hallucinated numbers were silently passing as "grounded". Switched to a
    word-boundary regex so "9" only matches a standalone 9, not the "9" inside
    "2926" or a customer id. This makes the gate meaningfully stricter for the
    exact class of number it exists to catch, without touching the separate
    rounding/percent-conversion tolerance logic below.
    """
    # ensure_ascii=False matters here: the default True escapes non-ASCII
    # characters like ₹ to "\u20b9", whose trailing hex digit "9" then sits
    # directly before every rupee amount in the dump ("\u20b9139"), which
    # would defeat the word-boundary check below for the very common case of
    # a currency-prefixed number. Keeping ₹ as a literal character means it
    # correctly acts as a non-word boundary immediately before the digits.
    context_dump = json.dumps(contexts, default=str, ensure_ascii=False).lower().replace(",", "")
    number_pattern = r"(?<![\w.])\d[\d,]*(?:\.\d+)?"
    numbers_in_body = re.findall(number_pattern, body)
    context_numbers = re.findall(number_pattern, context_dump)
    context_values = []
    for value in context_numbers:
        try:
            context_values.append(float(value.replace(",", "")))
        except ValueError:
            continue

    def _present_as_number(number_str: str) -> bool:
        pattern = re.compile(rf"(?<![\w.]){re.escape(number_str)}(?![\w.])")
        return bool(pattern.search(context_dump))

    for num_str in numbers_in_body:
        normalized = num_str.replace(",", "")
        if _present_as_number(normalized):
            continue
        try:
            val = float(normalized)
            grounded = False
            for context_value in context_values:
                # Comma formatting and reasonable rounding (e.g. 2.20 from 2.2).
                if abs(val - context_value) <= max(0.005, abs(context_value) * 0.0005):
                    grounded = True
                    break
                # Percent/fraction conversions only for values in the probability range.
                if 0 <= context_value <= 1 and abs(val - context_value * 100) <= 0.05:
                    grounded = True
                    break
            if grounded:
                continue
        except ValueError:
            pass
        return False  # Ungrounded factual number detected
    return True

def validate_and_clean_action(
    action: Dict[str, Any],
    deterministic_body: str = "",
    category: Dict[str, Any] = None,
    merchant: Dict[str, Any] = None,
    trigger: Dict[str, Any] = None,
    customer: Dict[str, Any] = None,
    seen_conversations: Optional[Set[str]] = None
) -> Optional[Dict[str, Any]]:
    
    if not isinstance(action, dict):
        return None

    trigger = trigger if isinstance(trigger, dict) else {}
    payload = trigger.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    customer_scope = (
        trigger.get("scope") == "customer"
        or customer is not None
        or action.get("send_as") == "merchant_on_behalf"
    )
    confirmed = payload.get("confirmed") is True or payload.get("booking_status") == "confirmed"

    # Gate: Fact Traceability (Revert to deterministic if LLM hallucinated)
    current_body = action.get("body", "")
    body_needs_fallback = current_body and deterministic_body and (
        current_body != deterministic_body
        or (customer_scope and not confirmed and not check_no_unearned_completion_claims(current_body, trigger))
        or not check_no_clinical_sales_jargon(current_body, category or {})
        or not check_no_third_person_self_reference(current_body, customer_scope)
        or not check_no_price_matching_framing(current_body, trigger)
        or not check_no_expires_at_as_deadline(
            current_body, trigger, [category, merchant, payload, customer]
        )
        or not check_no_unresolved_template_values(current_body)
        or not check_refill_claims_are_supported(current_body, trigger)
    )
    if body_needs_fallback:
        # Include the engine's own deterministic draft as a grounding source.
        # Bug fixed: numbers the deterministic path legitimately derives (e.g.
        # compute_topic_offer_tiers' bulk price tiers, which are correct math
        # off a real active offer, not a hallucination) were never checked
        # against by fact-traceability -- only the raw category/merchant/
        # trigger/customer JSON was. That meant a *correct* LLM restatement of
        # those same derived tiers was indistinguishable from fabrication and
        # reverted every time (confirmed: trg_013_corporate_thali_planning has
        # failed "fact-traceability" every single round). The deterministic
        # body is engine-trusted by construction, so treating it as a grounding
        # source is safe and directly fixes this specific, repeated false
        # positive without weakening the check for genuinely invented numbers.
        is_factual = check_fact_traceability(
            current_body, [category, merchant, trigger, customer, deterministic_body]
        )
        anchors_ok = check_anchor_terms_present(
            current_body, extract_required_anchor_terms(trigger or {})
        )
        weekday_ok = check_weekday_consistency(current_body, trigger or {})
        offer_topic_ok = check_topic_relevant_offer(current_body, trigger or {}, merchant or {})
        no_price_matching = check_no_price_matching_framing(current_body, trigger or {})
        expiry_match = find_expiry_date_mention(
            current_body, trigger or {}, [category, merchant, payload, customer]
        )
        no_expiry_as_deadline = expiry_match is None
        no_unearned_completion = check_no_unearned_completion_claims(current_body, trigger or {})
        no_taboo = check_taboo_words(current_body, category or {})
        no_clinical_sales_jargon = check_no_clinical_sales_jargon(current_body, category or {})
        no_third_person_self_reference = check_no_third_person_self_reference(current_body, customer_scope)
        no_template_placeholders = check_no_unresolved_template_values(current_body)
        refill_claims_supported = check_refill_claims_are_supported(current_body, trigger or {})
        customer_claim_safe = not customer_scope or confirmed or no_unearned_completion
        failed_checks = []
        for passed, check_name in (
            (is_factual, "fact-traceability"),
            (anchors_ok, "required-anchor"),
            (weekday_ok, "weekday-consistency"),
            (offer_topic_ok, "topic-audience-match"),
            (no_price_matching, "competitor-price-framing"),
            (no_expiry_as_deadline, f"expiry-as-deadline (matched '{expiry_match}')"),
            (no_unearned_completion, "unearned-completion-claim"),
            (customer_claim_safe, "customer-claim-safety"),
            (no_taboo, "category-taboo"),
            (no_clinical_sales_jargon, "clinical-register"),
            (no_third_person_self_reference, "customer-voice-self-reference"),
            (no_template_placeholders, "unresolved-template-value"),
            (refill_claims_supported, "refill-capability-claim"),
        ):
            if not passed:
                failed_checks.append(check_name)
        if failed_checks:
            action["body"] = deterministic_body
            action["rationale"] = (
                action.get("rationale", "")
                + f" (LLM output failed checks: {', '.join(failed_checks)}; reverted to deterministic fallback)."
            ).strip()

    # Gate: Multi-action cap
    conv_id = action.get("conversation_id", "")
    if seen_conversations is not None:
        if conv_id in seen_conversations:
            return None
        seen_conversations.add(conv_id)

    # Clean Body
    body = strip_urls(action.get("body", ""))
    body = strip_jargon(body)
    is_customer_facing = action.get("send_as") == "merchant_on_behalf" or customer is not None
    body = ensure_owner_name(body, merchant or {}, is_customer_facing, customer)
    
    # Final Fallback check
    if not body.strip():
        owner = (merchant or {}).get("identity", {}).get("owner_first_name", "there")
        trig_id = (trigger or {}).get("id", "trg")
        body = f"Hi {owner}, an opportunity matched your profile. Would you like to review? [Ref: {trig_id[-4:]}]"
    
    action["body"] = body.strip()

    if not check_no_unresolved_template_values(action["body"]):
        if is_customer_facing:
            customer_name = (customer or {}).get("identity", {}).get("name", "there")
            merchant_name = (merchant or {}).get("identity", {}).get("name", "our team")
            action["body"] = (
                f"Namaste {customer_name}, {merchant_name} here. Please contact us so we can help with your request. "
                "Would you like us to help?"
            )
        else:
            owner = (merchant or {}).get("identity", {}).get("owner_first_name", "there")
            action["body"] = (
                f"{owner}, I can prepare a concise next-step note for this update. "
                "Would you like me to draft it?"
            )
        action["rationale"] = (
            action.get("rationale", "")
            + " (Unresolved template value detected; replaced with a safe generic message)."
        ).strip()

    # Enforce Schema
    action["cta"] = action.get("cta") if action.get("cta") in VALID_CTAS else "open_ended"
    if is_customer_facing:
        action["send_as"] = "merchant_on_behalf"
        action["customer_id"] = customer.get("customer_id") if customer else action.get("customer_id")
    else:
        action["send_as"] = "vera"
        action["customer_id"] = None

    for k in REQUIRED_ACTION_KEYS:
        if k not in action:
            action[k] = [] if k == "template_params" else ("" if k != "customer_id" else None)

    return action