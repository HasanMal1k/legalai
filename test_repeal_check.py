"""Sentences the repeal check must accept and must refuse (taken from the corpus)."""
import sys

sys.path.insert(0, r"D:\Hasan\Scrapper")
import finalize_metadata as f

CASES = [
    # (expected, title, text)
    (True, "The Punjab Khal Panchayat Act, 2019", "2. Repeal and saving. (1) The Punjab Khal Panchayat Act 2019 (XXV of 2019) is hereby repealed."),
    (True, "The Punjab Local Government Act, 2019", "216. Repeal and Savings.- (1) The Punjab Local Government Act, 2019 (XIII of 2019) and the Village Panchayats and\nNeighbourhood Councils Act, 2019 (XIV of 2019) are hereby repealed."),
    (True, "The Village Panchayats and Neighbourhood Councils Act, 2019", "216. Repeal and Savings.- (1) The Punjab Local Government Act, 2019 (XIII of 2019) and the Village Panchayats and\nNeighbourhood Councils Act, 2019 (XIV of 2019) are hereby repealed."),
    (True, "The Representation of the People Act, 1976", "241. Repeal.—The following laws are hereby repealed—\n(a) the Electoral Rolls Act, 1974 (Act No. XXI of 1974);\n(b) the Delimitation of Constituencies Act, 1974 (Act No. XXXIV of\n1974);\n(c) the Senate (Election) Act, 1975 (Act No. LI of 1975);\n(d) the Representation of the People Act, 1976 (Act No. LXXXV of\n1976);\n(e) the Election Commission Order, 2002"),
    (True, "The Electoral Rolls Act, 1974", "241. Repeal.—The following laws are hereby repealed—\n(a) the Electoral Rolls Act, 1974 (Act No. XXI of 1974);\n(b) the Delimitation"),
    (True, "The Special Courts for Speedy Trials Act, 1992", "2. Repeal and savings.—(1) The Special Courts for Speedy Trials Act, 1992 (IX of 1992), hereinafter referred to as the said Act, shall stand repealed from the twenty-sixth day of July, 1994."),
    (True, "The Competition (Exemption) Regulations, 2014", "12. Repeal—On coming into force of these Regulations, the Competition (Exemption) Regulations, 2014 shall stand repealed."),
    (True, "The Balochistan Building and Town Planning Rules, 1979", "The Balochistan Building and Town Planning Rules 1979 and Building Code for Quetta Municipality, 1937 are hereby repealed."),
    (True, "The Salary of Judges of Superior Courts Order, 2007", "5. Repeal.— The salary of Judges of Superior Courts Order, 2007 (P.O.No.1 of 2007), is hereby repealed."),
    (True, "The Private Wildlife Breeding Farm Rules, 2008", "Repeal.- The Private Wildlife Breeding Farm Rules, 2008 are hereby repealed."),
    (False, "The General Clauses Act, 1897", "Any Gwadur law is hereby repealed and shall be deemed to have been re-enacted by such Federal law, as if the Gwadur law were a Federal Act, and the provisions of the General Clauses Act, 1897 (X of 1897), shall apply accordingly."),
    (False, "The Federal Excise Act, 2005", "(50) in the Federal Excise Act, 2005, the words shall be substituted; (51) the Second Schedule is hereby repealed; and (52) the amendments set out in the Second Schedule shall be made"),
    (False, "The Transfer of Property Act, 1882", "The words “in writing” are repealed wherever the Transfer of Property Act, 1882 (IV of 1882) is in force, see ss. 1 and 2."),
    (False, "The Balochistan Public Safety Regulation, 1947", "Sections 3, 4, 6, 7, 12, 17, 21, 22, 28, 29 and 30 of the Balochistan Public Safety Regulation, 1947, and sections 3, 4 of the Balochistan States Union Public Safety Regulation are hereby repealed."),
    (False, "The University of Karachi Act, 1972", "Clauses (f) and (g) of the University of Karachi Act, 1972 are hereby repealed."),
    (False, "The Trade Organisations Ordinance, 1961", "35. Repeal and savings.- On commencement of this Act,-- (a) the Trade Organisations Ordinance, 1961 (XLV of 1961), in its\napplication to trade organisations with objects not confined to one Province, shall stand repealed with effect from the 30th December, 2006;"),
    (False, "The Punjab Land Revenue Act, 1967", "in the Punjab Acquisition of Land (Housing) (Repeal) Act, 1985 (XII of 1985), in section 3, in the third proviso– (a) in paragraph 2, for entries (ii) and (iii), the following shall be substituted: the Punjab Land Revenue Act, 1967"),
    (False, "The Kazis Act, 1880", "The Kazis Act, 1880 is hereby repealed to the extent of the Islamabad Capital Territory."),
    (False, "The Factories Act, 1934", "The Factories Act, 1934 (XXV of 1934), in its application to the Province, is hereby repealed."),
    (False, "The Canal and Drainage Act, 1873", "The enactments specified in the Schedule are hereby repealed to the extent mentioned in the fourth column thereof. THE SCHEDULE: the Canal and Drainage Act, 1873, section 2"),
    (False, "The Contract Act, 1872", "The X Act, 1960 is hereby repealed. 5. Savings. Nothing in the Contract Act, 1872 shall apply."),
    (False, "The Arms Act, 1878", "The Explosives Act shall continue to apply, and the Arms Act, 1878 shall also apply; the Y Ordinance is hereby repealed."),
]
AVIATION = ("237. Repeal.- On the commencement of this Act and the Pakistan Civil Aviation Act, 2023, the Civil Aviation Ordinance, 1960 "
            "(XXXII of 1960) and the Pakistan Civil Aviation Authority Ordinance, 1982 (XXX of 1982) shall be repealed.")
LONG_LIST = ("80. Repeal.- (1) The following laws are hereby repealed: " + " ".join(
    f"({chr(97 + n)}) the Punjab Sample Law Number {n} Act {1990 + n} ({n} of {1990 + n});" for n in range(22))
    + " (w) the Punjab Occupational Safety and Health Act 2019 (IV of 2019); (x) the Punjab Minimum Wages Act 2019 (XXVIII of 2019). (2) Notwithstanding such repeal")
BAR = ("57. Repeal.- The following laws shall stand repealed, namely:— (1) The whole of the Legal Practitioners Act, 1879 (XVIII of 1879), "
       "except sections 1, 3 and 36. (2) The whole of the Legal Practitioners (Women) Act, 1923 (XXIII of 1923). (3) The whole of the Bar Councils Act, 1926")
CASES += [
    (True, "The Medical Diplomas Act, 1939", "1 Short title—This Act may be called the Medical Diplomas (Repeal) Act, 1966. 2 Repeal_The Medical Diplomas Act 1939 (XXVIII of 1939) is hereby repealed."),
    (True, "The Companies (E-Voting) Regulations, 2016", "12. Repeal.- Companies (E-Voting) Regulations, 2016 issued vide S.R.O 43 (I)/2016 dated January 22, 2016 is hereby repealed. Page 6 of 9"),
    (True, "The Contempt of Court Act, 1976", "(2) For removal of doubt it is hereby declared that the Contempt of Court\nAct, 1976 (LXIV of 1976), the Contempt of Court Ordinance, 2003 (V of 2003) and the Contempt of Court Ordinance, 2004 (I of 2004) stand repealed"),
    (True, "The Contempt of Court Ordinance, 2004", "(2) For removal of doubt it is hereby declared that the Contempt of Court\nAct, 1976 (LXIV of 1976), the Contempt of Court Ordinance, 2003 (V of 2003) and the Contempt of Court Ordinance, 2004 (I of 2004) stand repealed"),
    (True, "The Employees' Social Insurance Ordinance, 1962", "82. Repeal.– The Employees’ Social Insurance Ordinance, 1962 (XXII of 1962), in its application to the Province of West Pakistan, is hereby repealed."),
    (True, "The Sindh Public Service Commission Act, 1989", "(1) The Sindh Public Service Commission Act, 1989, is hereby Repeal and savings repealed."),
    (True, "The Civil Laws (Reforms) Ordinance, 1994", "Repeal The Civil Laws (Reforms) Ordinance, 1994 (VI of 1994), is hereby repealed"),
    (True, "The Kalri Lake Development Ordinance, 1959", "2. Repeal.– The Kalri Lake Development Ordinance, 19591, is hereby repealed."),
    (True, "The Undesirable Companies Ordinance, 1957", "13.-(1) The Undesirable Companies Ordinance, 1957,21 is hereby repealed"),
    (True, "The Pakistan Civil Aviation Authority Ordinance, 1982", AVIATION),
    (True, "The Civil Aviation Ordinance, 1960", AVIATION),
    (False, "The Pakistan Civil Aviation Act, 2023", AVIATION),
    (True, "The National Sports Trust Act, 1972", "ORDINANCE No. XXXII of 1980 AN ORDINANCE to repeal the National Sports Trust Act, 1972 Whereas it is expedient", {"purpose": True}),
    (False, "The National Sports Trust Act, 1972", "ORDINANCE No. XXXII of 1980 AN ORDINANCE to repeal the National Sports Trust Act, 1972 Whereas it is expedient"),
    (True, "The Punjab Occupational Safety and Health Act, 2019", LONG_LIST),
    (True, "The Legal Practitioners (Women) Act, 1923", BAR),
    (False, "The Legal Practitioners Act, 1879", BAR),
    (False, "The Defence of Pakistan Ordinance, 1971", "in relation to land which, when the Defence of Pakistan Ordinance, 1971 (XXX of 1971) stands repealed, is subject to any requisition"),
    (False, "The Lahore Development Authority Act, 1975", "(1) Sub-sections (2) to (6) of section 10, sub-section (2) of section 28 and section 29 of the Lahore Development Authority Act, 1975 (XXX of 1975), and clause (xvi) of sub-section (2) of section 7 of the Punjab Development of Cities Act, 1976 (XIX of 1976) are hereby repealed"),
    (False, "The Punjab Development of Cities Act, 1976", "(1) Sub-sections (2) to (6) of section 10, sub-section (2) of section 28 and section 29 of the Lahore Development Authority Act, 1975 (XXX of 1975), and clause (xvi) of sub-section (2) of section 7 of the Punjab Development of Cities Act, 1976 (XIX of 1976) are hereby repealed"),
    (False, "The Sample Rules, 2001", "The Sample Rules, 2001 made under section 5 apply. The Other Rules, 2010 are hereby repealed."),
    (True, "The Arbitration (Protocol and Convention) Act, 1937", "10. Repeal and saving.⸻(1) The Arbitration (Protocol and Convention) Act, 1937 (VI of 1937) (hereinafter in this section referred to as “the Act”) is hereby repealed. (2) Notwithstanding"),
    (True, "The Karachi Indecent Advertisement Prohibition Act, 1951", "14, -Repeals.—The Karachi Indecent Advertisement Prohibition Act, 1951 (XXXVI of 1951), and the Punjab Suppression of Indecent Adver- tisements Act, 1941 (Punjab Act VII of 1941), are hereby repealed."),
    (True, "The West Punjab Cotton Control Act, 1949", "31. Repeal and savings.– (1) The West Punjab Cotton Control Act, 1949 (IV of 1949), 3[the Bahawalpur Cotton Control Act, 1949, the Cotton Act, 1952 (Khairpur Act V of 1952)], and the Sind Cotton Control Act, 1954 (I of 1954), are hereby repealed."),
    (True, "The Sindh Public Service Commission Act, 1989", "14. (1) The Sindh Public Service Commission Act, 1989, is hereby Repeal and saving. repealed. (2) Notwithstanding the repeal"),
    (True, "The Sindh Disposal of Plots Ordinance, 1980", "18. The Sindh Disposal of Plots Ordinance, 1980 is hereby Repeal. repealed."),
    (True, "The Sindh Agricultural Income Tax Ordinance, 2000", "23. (1) The Sindh Agricultural Income Tax Ordinance, 2000 shall Repeal and stand repealed with effect from the date this Act comes into Saving. force."),
    (False, "The West Pakistan Weights and Measures Ordinance, 1965", "as may be notified by Government the provisions of West Pakistan Weights and Measures Ordinance, 1965 shall stand repealed to that extent."),
    (False, "The Factories Act, 1934", "71. Savings.- The following laws shall remain in force till the time the rules are made: (a) The Factories Act, 1934; and (b) The Punjab Occupational Safety and Health Act 2019."),
]
bad = 0
for case in CASES:
    expected, title, text = case[:3]
    got = f.repeal_in_text(text, title, **(case[3] if len(case) > 3 else {}))
    ok = bool(got) == expected
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} expect {'repeal' if expected else 'none  '} | {title[:48]:48s} | {str(got)[:90]}")

# a stale copy's whole repeal, limited by the repealing law as amended on an up-to-date code (limited_on_code): the
# Elections Act, 2017 on DRS (as first published) and on the Pakistan Code (with the exception inserted in 2017)
ELECTIONS_FIRST = ("241. Repeal.—The following laws are hereby repealed— (e) the Election Commission Order, 2002 (Chief Executive’s Order "
                   "No.1 of 2002); (f) the Conduct of General Elections Order, 2002 (Chief Executive’s Order No.7 of 2002); (g) the Political "
                   "Parties Order, 2002 (Chief Executive’s Order No.18 of 2002); and (h) the Allocation of Symbols Order, 2002.")
ELECTIONS_NOW = ELECTIONS_FIRST.replace("No.7 of 2002);", "No.7 of 2002)1[******] 2[except Articles 1, 7B and 7C];")


def copy(source, text):
    return {"title": "The Elections Act, 2017", "source": source, "jurisdiction": "Federal", "_doc": {"text": text}}


first, now = copy("drs", ELECTIONS_FIRST), copy("pakistancode", ELECTIONS_NOW)
for title, copies, of, want in [("The Conduct of General Elections Order, 2002", [first, now], first, True),
                                ("The Political Parties Order, 2002", [first, now], first, False),
                                # the code's own copy is the law as amended: nothing to limit
                                ("The Conduct of General Elections Order, 2002", [first, now], now, False),
                                # two gazette copies: no up-to-date text to go by
                                ("The Conduct of General Elections Order, 2002", [first, copy("drs", ELECTIONS_NOW)], first, False)]:
    got = f.limited_on_code(of, title, copies)
    ok = bool(got) == want
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} expect {'limited' if want else 'whole  '} | {title[:40]:40s} read in {of['source']:12s} | {str(got)[:80]}")
print("failures:", bad)
sys.exit(1 if bad else 0)
