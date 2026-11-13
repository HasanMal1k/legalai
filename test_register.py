"""The register of laws in force (register.py -> data/register/register.jsonl) must get the major laws right:
canaries whose status is settled law, and invariants no build may break.

Run: .venv\\Scripts\\python.exe test_register.py   (exits 1 on any failure)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).parent / "data" / "register"
IN_FORCE = {"in force", "in force (in part)", "in force (evidence)"}

# (jurisdiction, title pattern, year, the levels it may have)
SINDH_OK = IN_FORCE | {"not confirmed"}
CANARIES = [(where, pattern, year, IN_FORCE) for where, pattern, year in [
    ("Federal", r"Pakistan Penal Code", 1860), ("Federal", r"Code of Criminal Procedure", 1898), ("Federal", r"Code of Civil Procedure", 1908),
    ("Federal", r"^The Contract Act", 1872), ("Federal", r"Qanun-e-Shahadat", 1984), ("Federal", r"^The Companies Act", 2017),
    ("Federal", r"Income Tax Ordinance", 2001), ("Federal", r"Sales Tax Act", 1990), ("Federal", r"^The Customs Act", 1969),
    ("Federal", r"^The Limitation Act", 1908), ("Federal", r"^The Specific Relief Act", 1877), ("Federal", r"^The Transfer of Property Act", 1882),
    ("Federal", r"^The Registration Act", 1908), ("Federal", r"^The Stamp Act", 1899), ("Federal", r"^The Succession Act", 1925),
    ("Federal", r"Guardians and Wards Act", 1890), ("Federal", r"Muslim Family Laws Ordinance", 1961),
    ("Federal", r"Dissolution of Muslim Marriages", 1939), ("Federal", r"Anti-?Terrorism Act", 1997), ("Federal", r"National Accountability", 1999),
    ("Federal", r"Prevention of Electronic Crimes Act", 2016), ("Federal", r"^The Arbitration Act", 1940), ("Federal", r"^The Partnership Act", 1932),
    ("Federal", r"^The Sale of Goods Act", 1930), ("Federal", r"Negotiable Instruments Act", 1881),
    ("Federal", r"Control of Narcotic Substances Act", 1997), ("Federal", r"^The Elections Act", 2017), ("Federal", r"^The Civil Servants Act", 1973),
    ("Federal", r"Industrial Relations Act", 2012), ("Federal", r"^The Federal Excise Act", 2005), ("Federal", r"^The Insurance Ordinance", 2000),
    ("Federal", r"^The Securities Act", 2015), ("Federal", r"Banking Companies Ordinance", 1962), ("Federal", r"State Bank of Pakistan Act", 1956),
    ("Federal", r"Recovery of Finances", 2001), ("Federal", r"^The Copyright Ordinance", 1962), ("Federal", r"^The Patents Ordinance", 2000),
    ("Federal", r"^The Competition Act", 2010), ("Federal", r"^The General Clauses Act", 1897), ("Federal", r"Juvenile Justice System Act", 2018),
    ("Federal", r"Transgender Persons", 2018), ("Federal", r"Federal Investigation Agency Act", 1974), ("Federal", r"^The Oaths Act", 1873),
    ("Federal", r"Legal Practitioners and Bar Councils Act", 1973), ("Federal", r"Electronic Transactions Ordinance", 2002),
    ("Federal", r"^The Majority Act", 1875), ("Federal", r"^The Christian Marriage Act", 1872), ("Federal", r"^The Trade Marks Ordinance", 2001),
    ("Punjab", r"Punjab Land Revenue Act", 1967), ("Punjab", r"Protection of Women against Violence Act", 2016),
    ("Punjab", r"Punjab Consumer Protection Act", 2005), ("Khyber Pakhtunkhwa", r"Local Government Act", 2013),
    ("Khyber Pakhtunkhwa", r"Right to Information Act", 2013), ("Balochistan", r"Balochistan Local Government Act", 2010)]]
CANARIES += [(where, pattern, year, {"not in force"}) for where, pattern, year in [
    ("Federal", r"^The Companies Ordinance", 1984), ("Federal", r"^The Ehtesab Act", 1997), ("Federal", r"Industrial Relations Ordinance", 2002),
    ("Federal", r"Industrial Relations Act", 2008), ("Federal", r"Industrial Relations Ordinance", 1969),
    ("Federal", r"Securities and Exchange Ordinance", 1969), ("Federal", r"^The Insurance Act", 1938),
    ("Federal", r"Pakistan Medical Commission Act", 2020), ("Federal", r"Freedom of Information Ordinance", 2002),
    ("Federal", r"Political Parties Act", 1962), ("Federal", r"Representation of (the )?People Act", 1976), ("Federal", r"Wealth-?Tax Act", 1963),
    ("Federal", r"^The Trade Marks Act", 1940), ("Federal", r"Patents and Designs Act", 1911),
    ("Federal", r"Juvenile Justice System Ordinance", 2000), ("Federal", r"Pakistan Medical and Dental Council Ordinance", 1962),
    # repealed for the Islamabad Capital Territory and every province, one by one, in 2020 (kg.py rule R10)
    ("Federal", r"^The (?:Indian )?Trusts? Act", 1882),
    ("Punjab", r"^The Punjab Local Government Act", 2013),
    ("Balochistan", r"Subordinate Judiciary Service Tribunal Act", 1989)]]
CANARIES += [("Sindh", r"Rented Premises Ordinance", 1979, SINDH_OK), ("Sindh", r"Sindh Local Government Act", 2013, SINDH_OK)]

register = [json.loads(line) for line in (OUT / "register.jsonl").open(encoding="utf-8")]
bad = 0


def check(ok: bool, line: str) -> None:
    global bad
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {line}")


for where, pattern, year, allowed in CANARIES:
    # (the year on file, or the year the name prints: "The Federal Investigation Agency Act, 1974" is filed under 1975)
    # (and the law itself, not a notification or regulation made under it, whose name ends with the law's)
    hits = [e for e in register if e["jurisdiction"] == where and re.search(pattern, e["title"], re.I)
            and str(year) in (str(e["year"]), (re.findall(r"(?:1[89]|20)\d{2}", str(e["title"])) or [None])[-1])
            and not re.search(r"Notification|Regulation|Rules", str(e["kind"]))]
    levels = sorted({e["level"] for e in hits})
    check(bool(hits) and set(levels) <= allowed, f"{where[:12]:12} {pattern[:45]:45} {year}: {levels or 'missing'}")

# one federal law, one entry: the same words (no "the", "act", "Indian"), the same kind and year
decoration = {"of", "and", "the", "for", "na", "act", "bill", "royal", "indian"}
seen = defaultdict(list)
for e in register:
    if e["jurisdiction"] == "Federal":
        name = re.sub(r"[,\s]*\(?(?:1[89]|20)\d{2}\)?\s*$", "", re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", str(e["title"])))
        words = frozenset(w for w in re.sub(r"[^a-z0-9]+", " ", name.lower()).split() if w not in decoration)
        printed = re.search(r"((?:1[89]|20)\d{2})\W*$", str(e["title"]))      # (the Presidential Cabinet Orders of 1971 and 1972 are two)
        seen[(words, str(e["kind"]), str(e["year"]), printed.group(1) if printed else None)].append(e)
twice = [v for v in seen.values() if len(v) > 1 and len({e["level"] for e in v}) > 1]
check(not twice, f"no federal law listed twice with two levels ({len(twice)} found)" +
      "".join(f"\n       {' / '.join(e['title'] + ': ' + e['level'] for e in v)}" for v in twice[:10]))

# the Pakistan Code's list marks some laws "(Repealed by Act XIX of 2017)": none of them in force
noted = [e for e in register if any(x.startswith("the Pakistan Code's list names it") and re.search(r"\(\s*(?:repealed|repeal|rep\.)\s+by", x, re.I)
                                    and not re.search(r"\bvalid\b|\bup\s*to\b|when notified", x, re.I) for x in e["evidence"])]
wrong = [e for e in noted if e["level"] in IN_FORCE | {"probably in force"}]
check(bool(noted) and not wrong, f"{len(noted)} laws the Pakistan Code's list marks repealed; at an in-force level: {[e['title'] for e in wrong]}")

drafts = [e for e in register if re.match(r"\W*draft\b", e["title"], re.I) and e["level"] in ("in force", "in force (evidence)")]
check(not drafts, f"no draft at an in-force level ({[e['title'] for e in drafts]})")

print("failures:", bad)
sys.exit(1 if bad else 0)
