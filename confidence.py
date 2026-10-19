"""
confidence.py

How far each law in the index can be trusted, and therefore where it may be served from.
Called by finalize_metadata.py after deduplication and the knowledge graph (kg.py); adds to every row:

    confidence   {"text": high|medium|low, "status": verified|probable|unverified,
                  "identity": verified|probable|conflict, "reasons": [...]}
    tier         current | unconfirmed | historical | quarantine   (None for documents outside the index)
    quarantine_reasons   why a law is held back, when it is

The rules, strict on purpose -- a law is served only on positive evidence:

  text      high: under 2% of its ordinary words garbled (seen in fewer than 5 documents);
            medium: under 6%; low: more, or most pages without any text.
  status    verified in force: listed as current on the official code of the jurisdiction the
            law belongs to (the Pakistan Code for federal law, a province's code for its own and
            for the West Pakistan laws it inherited), with no repeal or open lead against it
            anywhere in the corpus. A federal law on a province's code is verified only when the
            Pakistan Code lists it as current too -- or, for an Act or Ordinance, when the Pakistan
            Code does not carry it at all (the province's code is then its only authority). Rules
            made under an Act that has been repealed are not verified.
            verified repealed: the site says so, or the repealing law's own words do (checked by
            finalize_metadata.repeal_in_text and kg.py). Anything weaker is probable or unverified.
  identity  verified: the law's own short-title clause gives the same title, or the document
            prints the title at its head and the site's name for it agrees. conflict: the passes
            or the site disagree on title or kind, and the text settles nothing.

  current      in force, status verified, text not low, identity not in conflict
  unconfirmed  a Sindh law nothing repeals or contradicts, text not low, identity not in conflict: served with
               the label "status not confirmed", since the Sindh Code does not say which laws are current
  historical   repealed, superseded or lapsed; status verified or probable; text not low. Not mixed into
               answers: shown as "what was the law before" (chunks carry replaces / replaced_by)
  quarantine   everything else in the index: kept, never served, each with reasons
"""

import re
from collections import Counter

from rapidfuzz import fuzz

PROVINCE_CODES = {"punjab", "kp", "balochistan"}      # codes that say what is current (Sindh's site does not)
REPEAL_PROOF = ("the website's entry", "listed under repealed laws", "note at the head", "repealed by ")
LEADS = ("possibly_affected_by", "repealed_federally", "possibly_repealed_by")
KIND_WORD = re.compile(r"\b(?:act|ordinance|order|rules|regulations?|code)\b")


def text_level(record: dict) -> tuple[str, str | None]:
    share = record["garbled_words"]
    pages, chars = record.get("num_pages") or 0, len(record["_doc"]["text"])
    if record["text_source"] == "machine-translation":
        return "medium", "a machine translation of an Urdu text"
    if pages >= 3 and chars < 300 * pages:
        return "low", "most pages have no text"
    if share <= 0.02:
        return "high", None
    # its uncommon words were read the same way by a second reader of the scan (repair_pages.py): real words, not misreadings
    check = (record.get("text_check") or {}).get("verdict")
    if check == "confirmed":
        return "high", None
    if check == "confirmed_by_sample":
        return "high", "the worst pages were read twice and agreed; the rest were not re-read"
    if share <= 0.06:
        return "medium", f"{share:.0%} of ordinary words garbled"
    return "low", f"{share:.0%} of ordinary words garbled"


def listed(record: dict) -> bool:
    """Does an official code vouch for the law as current: in its listing of current law, or by keeping its text
    amended up to recent years (kg.kept_current)?"""
    return bool(record["status_basis"]) and record["status_basis"][0].startswith(("listed as current law on the",
                                                                                  "kept up to date on the Pakistan Code"))


def status_level(record: dict, federal_current: set, federal_any: set = frozenset()) -> tuple[str, str | None]:
    status = record["status"]
    basis = record["status_basis"][0] if record["status_basis"] else ""
    if status == "repealed":
        return ("verified", None) if basis.startswith(REPEAL_PROOF) else ("probable", "repeal shown only by a note or a repeal that could not be fully confirmed")
    if status == "in_force":
        own_code = record["source"] == "pakistancode" or (record["source"] in PROVINCE_CODES and record["jurisdiction"] != "Federal")
        leads = [x for x in record.get("related", []) if x["relation"] in LEADS]
        if not listed(record):
            return "probable", "in force by inference, not listed as current"
        if leads:
            return "probable", "another law names it in a repeal that could not be confirmed"
        parent = next((x for x in record.get("related", []) if x["relation"] == "parent_repealed"), None)
        if parent:
            # rules and regulations fall with the Act they were made under, unless the repealing law saved them
            return "probable", f"made under {parent['title']}, which has since been repealed"
        if not own_code:
            if record["_norm"] in federal_current or record.get("family_id") in federal_current:
                return "verified", None
            # an Act or Ordinance the Pakistan Code does not carry at all: the province's code is the authority for it (its
            # own law, inherited as a central Act, or a devolved subject) -- unlike rules, which the Pakistan Code never hosts
            if record["instrument_type"] in ("Act", "Ordinance", "Order", "Regulation") and record["_norm"] not in federal_any \
                    and record.get("family_id") not in federal_any:
                return "verified", None
            return "probable", "a federal law as a provincial code lists it (the Pakistan Code does not list it as current)"
        return "verified", None
    if status == "superseded":
        return "probable", "replaced by an Act of the same name (no repealing words found)"
    if status == "lapsed_presumed":
        return "probable", "an Ordinance presumed to have lapsed"
    if status == "unknown" and basis.startswith("conflict"):
        return "unverified", "the sources conflict: " + basis[len("conflict: "):][:160]
    return "unverified", {"unknown": "no evidence of whether it is in force", "amending_instrument": "an amending law"}.get(status, status)


def unconfirmed_sindh(record: dict) -> bool:
    """A Sindh law whose status nothing settles: the Sindh Code does not say which laws are current, and nothing in the
    corpus repeals it, conflicts with it or leaves an open lead against it. Served, labelled "status not confirmed"."""
    if record["source"] != "sindh" or record["status"] not in ("unknown", "in_force"):
        return False
    basis = record["status_basis"][0] if record["status_basis"] else ""
    return not basis.startswith("conflict") and not any(x["relation"] in LEADS for x in record.get("related", []))


def heading_shows(record: dict) -> bool:
    """Does the document print its own title near the top (where a law states its name)?"""
    text = record["_doc"]["text"]
    head = text[:3000]
    body = re.search(r"\nTEXT\s*\n", text[:20000])
    if body:
        head += " " + text[body.end():body.end() + 1500]
    title = re.sub(r"\b(?:1[89]|20)\d{2}\b", " ", record["_norm"])
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) < 8:
        return False
    flat = re.sub(r"[^a-z0-9 ]+", " ", head.lower())
    flat = re.sub(r"\s+", " ", re.sub(r"\bn w f p\b|\bnorth west frontier province\b", "khyber pakhtunkhwa", flat))
    year_ok = not record["year"] or str(record["year"]) in head
    # a law may print its name without its kind ("THE QANUN E SHAHADAT, 1984" heads the Qanun-e-Shahadat Order)
    kindless = re.sub(r"\s+", " ", KIND_WORD.sub(" ", title)).strip()
    return year_ok and (fuzz.partial_ratio(title, flat) >= 90 or (len(kindless) >= 8 and fuzz.partial_ratio(kindless, flat) >= 95))


def identity_level(record: dict, cited, same_title, norm_title) -> tuple[str, str | None]:
    if cited and same_title(norm_title(f"{cited[0]}, {cited[1]}"), record["_norm"]):
        return "verified", None
    # the site's name, without the abbreviation some entries end with ("Qanun-e-Shahadat Order, 1984 (QSO)")
    site_name = re.sub(r"\s*\([A-Z]{2,8}\)\s*$", "", record["name"])
    site_agrees = same_title(norm_title(site_name), record["_norm"])
    shows = heading_shows(record)
    if shows and site_agrees:
        return "verified", None
    if not record["needs_review"] or shows or site_agrees:
        return "probable", None if record["needs_review"] is False else "title confirmed by only one of the site and the text"
    return "conflict", "title or kind of law disputed between the site and the two model readings, and the text does not settle it"


def assign(records: list[dict], same_title, norm_title, own_short_title) -> dict:
    # federal laws the Pakistan Code lists as current, with nothing against them: what a provincial copy of
    # a federal law is checked against
    federal_current = set()
    for record in records:
        if record["source"] == "pakistancode" and record["status"] == "in_force" and listed(record) \
                and not any(x["relation"] in LEADS for x in record.get("related", [])):
            federal_current.add(record["_norm"])
            federal_current.add(record.get("family_id"))
    federal_current.discard(None)
    # every law the Pakistan Code carries, whatever its status there
    federal_any = {r["_norm"] for r in records if r["source"] == "pakistancode"} | \
                  {r.get("family_id") for r in records if r["source"] == "pakistancode"}
    federal_any.discard(None)

    tiers = Counter()
    for record in records:
        text, why_text = text_level(record)
        status, why_status = status_level(record, federal_current, federal_any)
        identity, why_identity = identity_level(record, own_short_title(record["_doc"]["text"]), same_title, norm_title)
        record["confidence"] = {"text": text, "status": status, "identity": identity,
                                "reasons": [why for why in (why_text, why_status, why_identity) if why]}
        if not record["index_default"]:
            record["tier"], record["quarantine_reasons"] = None, []
            continue
        held = []
        if text == "low":
            held.append(f"text: {why_text}")
        if identity == "conflict":
            held.append(f"identity: {why_identity}")
        if unconfirmed_sindh(record):
            # the user's decision (2026-10-18): Sindh laws are served, labelled "status not confirmed"
            tier = "unconfirmed"
        elif record["status"] == "in_force":
            tier = "current" if status == "verified" else None
            if status != "verified":
                held.append(f"status: {why_status}")
        elif record["status"] in ("repealed", "superseded", "lapsed_presumed"):
            tier = "historical"
        else:
            tier = None
            held.append(f"status: {why_status}")
        record["tier"] = "quarantine" if held or tier is None else tier
        record["quarantine_reasons"] = held
        tiers[record["tier"]] += 1
    return dict(tiers)
