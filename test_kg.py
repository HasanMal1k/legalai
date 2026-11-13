"""Sentences from the corpus that the knowledge graph must read right: which laws a clause repeals,
and how much of each (whole / part / territory / conditional). None = no repeal edge at all.

Run: .venv\\Scripts\\python.exe test_kg.py   (exits 1 on any failure)
"""
import re
import sys

sys.path.insert(0, r"D:\Hasan\Scrapper")
import finalize_metadata as f
import kg

sys.stdout.reconfigure(encoding="utf-8")


def law(title, source="pakistancode", jurisdiction="Federal", kind=None, number=None, text="", is_amendment=False, year=None):
    year = year or f.title_year(title)
    # the kind is the word the title ends with ("... and Regulations Act, 2025" is an Act)
    last = re.findall(r"\b(Act|Ordinance|Order|Rules|Regulations?|Code)\b", title)
    kind = kind or {"Regulations": "Regulation", "Code": "Act"}.get(last[-1], last[-1]) if last else "Act"
    return {"title": title, "name": title, "year": year, "number": number, "_norm": f.norm_title(title), "instrument_type": kind,
            "source": source, "jurisdiction": jurisdiction, "work_id": "w-" + title, "is_amendment": is_amendment,
            "_doc": {"text": text, "id": title}, "status": "in_force", "status_basis": ["listed as current law on the Pakistan Code"], "related": [],
            "garbled_words": 0.0}


# (repealer title, its source and jurisdiction, its text, {target title: expected extent or None}, target source/jurisdiction)
CASES = [
    ("The Islamabad Capital Territory Child Marriage Restraint Act, 2025", "pakistancode", "Federal",
     "13. Repeal and Savings.__ (1) The Child Marriage Restraint Act, 1929 is hereby repealed to the extent of Islamabad Capital Territory. Page 4 of 5 (2) Notwithstanding the repeal",
     {"The Child Marriage Restraint Act, 1929": "territory"}, ("pakistancode", "Federal")),
    ("The Islamabad Capital Territory Trust Act, 2020", "pakistancode", "Federal",
     "111. Repeal.— The Trust Act, 1882 (II of 1882) is hereby repealed to the extent of Islamabad Capital Territory, Islamabad. 112. Saving.",
     {"The Trusts Act, 1882": "territory"}, ("pakistancode", "Federal")),
    ("The Punjab Immovable Public Properties (Removal of Encroachment) Ordinance, 2022", "punjab", "Punjab",
     "17. Repeal and Saving.- (1) The Punjab Autonomous Bodies Immovable Property (Ejectment of Unauthorized Occupants) Ordinance, 1965 "
     "(XXXVII of 1965) and the Punjab Government Lands and Buildings (Recovery of Possession) Ordinance, 1966 (IX of 1966) are hereby repealed. "
     "(2) Notwithstanding the repeal",
     {"The Punjab Autonomous Bodies Immovable Property (Ejectment of Unauthorized Occupants) Ordinance, 1965": "whole",
      "The Punjab Government Lands and Buildings (Recovery of Possession) Ordinance, 1966": "whole"}, ("punjab", "Punjab")),
    ("The Punjab Protected Areas Act, 2020", "punjab", "Punjab",
     "34. Repeal and saving.− (1) Section 3 to the extent of honorary officers and sections 16, 17, 18 and 19 of the Punjab Wildlife "
     "(Protection, Preservation, Conservation and Management) Act, 1974 (II of 1974) are hereby repealed. (2) Notwithstanding",
     {"The Punjab Wildlife (Protection, Preservation, Conservation and Management) Act, 1974": "part"}, ("punjab", "Punjab")),
    ("The Punjab Horticulture Authority Act, 2025", "punjab", "Punjab",
     "41. Repeal, omission and savings. (1) The Parks and Horticulture Authority Act 2012 (XLVII of 2012), sub-sections (2) to (6) of section 10, "
     "sub-section (2) of section 28 and section 29 of the Lahore Development Authority Act, 1975 (XXX of 1975), and clause (xvi) of sub-section (2) "
     "of section 7 of the Punjab Development of Cities Act, 1976 (XIX of 1976) are hereby repealed.",
     {"The Parks and Horticulture Authority Act, 2012": "whole", "The Lahore Development Authority Act, 1975": "part",
      "The Punjab Development of Cities Act, 1976": "part"}, ("punjab", "Punjab")),
    ("The Khyber Pakhtunkhwa Infrastructure Development Cess Act, 2022", "kp", "Khyber Pakhtunkhwa",
     "22. Repeal and savings.---(1) Subject to the provisions of section 4 of the General Clauses Act, 1956 (Act No. VI of 1956), the Khyber "
     "Pakhtunkhwa Finance Act, 2013 (Khyber Pakhtunkhwa Act No. XXI of 2013), is hereby repealed to the extent of section 94 thereof.",
     {"The Khyber Pakhtunkhwa Finance Act, 2013": "part", "The General Clauses Act, 1956": None}, ("kp", "Khyber Pakhtunkhwa")),
    ("The Natural Gas Regulatory Authority Ordinance, 2002", "pakistancode", "Federal",
     "(2) The Natural Gas (Development Surcharge) Ordinance, 1967 (1 of 1967), and the Natural Gas (Development Surcharge) Rules, 1967, and the "
     "Natural Gas Rules, 1960, so far as their provisions are inconsistent with the pro- visions of this Ordinance are hereby repealed. 34.",
     {"The Natural Gas (Development Surcharge) Ordinance, 1967": "part"}, ("pakistancode", "Federal")),
    ("The Representation of the People Act, 1976", "pakistancode", "Federal",
     "109. Repeal. The National and Provincial Assemblies (Elections) Ordinance, 1970 (XIII of 1970), and the National and Provincial Assemblies "
     "(Elections to Reserved Seats) Act, 1976 (XVIII of 1976), are hereby repealed in their application to the elections to which this Act applies.",
     {"The National and Provincial Assemblies (Elections to Reserved Seats) Act, 1976": "part"}, ("pakistancode", "Federal")),
    ("The National Metrology Institute of Pakistan Act, 2022", "pakistancode", "Federal",
     "26. Repeal and savings.__ (1) The provisions of Weights and Measures (International System) Act, 1967 (V of 1967) and any other law for "
     "the time being in force to the extent of the matters covered under this Act shall stand repealed. (2) Upon",
     {"The Weights and Measures (International System) Act, 1967": "part"}, ("pakistancode", "Federal")),
    ("The Khyber Pakhtunkhwa Private Schools Regulatory Authority Act, 2017", "kp", "Khyber Pakhtunkhwa",
     "32. Repeal and Saving.---(1)The Khyber Pakhtunkhwa Registration and Functioning of Private Educational Institutions Ordinance, 2001 "
     "(Khyber Pakhtunkhwa Ord. No. XXVII of 2001) is hereby repealed to the extent of matter pertaining to the Elementary and Secondary Education",
     {"The Khyber Pakhtunkhwa Registration and Functioning of Private Educational Institutions Ordinance, 2001": "part"}, ("kp", "Khyber Pakhtunkhwa")),
    ("The Gomal Zam Project (Control and Prevention of Speculation in Land) Ordinance, 2001", "kp", "Khyber Pakhtunkhwa",
     "21. Repeal.---(1) The North-West Frontier Province Gomal Zam Project (Control and Prevention of Speculation in Land) Ordinance, 1971 "
     "(N.-W.F.P. Ord. No. VIII of 1971), is hereby repealed. (2) Notwithstanding",
     {"The Khyber Pakhtunkhwa Gomal Zam Project (Control and Prevention of Speculation in Land) Ordinance, 1971": "whole"}, ("kp", "Khyber Pakhtunkhwa")),
    ("The Balochistan Boards of Intermediate and Secondary Education Act, 2019", "balochistan", "Balochistan",
     "Repeal and 38. (1) The Balochistan Board of Intermediate and Savings. Secondary Education Ordinance, 1977 (Ordinance No. X of 1977) is "
     "hereby repealed. (2) Notwithstanding",
     {"The Balochistan Board of Intermediate and Secondary Education Ordinance, 1977": "whole"}, ("balochistan", "Balochistan")),
    ("The Balochistan Animal Slaughter Control Act, 2025", "balochistan", "Balochistan",
     "Repeal and 15. (1) The West Pakistan Animal Slaughter saving. Control Act, 19631 (Act No. Il of 1963) is here by repealed. (2) Notwithstanding",
     {"The West Pakistan Animals Slaughter Control Act, 1963": "whole"}, ("balochistan", "Balochistan")),
    ("The Balochistan Industrial Development and Regulations Act, 2025", "balochistan", "Balochistan",
     "Repeal and 24. (1) The Balochistan Industries (Control on savings. Establishment and Enlargement) Ordinance, 1963 (W.P. Ordinance No. IV "
     "of 1963) hereby stands repealed. (2) Save as otherwise provided",
     {"The Balochistan Industries (Control on Establishment and Enlargement) Ordinance, 1963": "whole"}, ("balochistan", "Balochistan")),
    ("The Balochistan Waqf Properties Act, 2020", "balochistan", "Balochistan",
     "Repeal and 33.(1) The Balochistan Waqf Properties Ordinance, 1979 (IV of Savings. 1979), on commencement of this Act, shall stand repealed. "
     "(2)Notwithstanding the repeal",
     {"The Balochistan Waqf Properties Ordinance, 1979": "whole"}, ("balochistan", "Balochistan")),
    ("The Balochistan Security of Vulnerable Establishments Act, 2024", "balochistan", "Balochistan",
     "Repeal and 22. (1) The Balochistan Shops and savings. Establishments (Security) Act, 2009 VI of 2009) is hereby repealed. (2) Notwithstanding",
     {"The Balochistan Shops and Establishments (Security) Act, 2009": "whole"}, ("balochistan", "Balochistan")),
    ("The Bolan University of Medical and Health Sciences Act, 2017", "balochistan", "Balochistan",
     "Repeal and savings. 51. (1) The Bolan Medical College (Board of Governors) Act, 1990 (Act III of 1990) and the Bolan University of Medical "
     "and Health Sciences Ordinance, 2007 (XVIII of 2007), hereinafter referred to as “the Repealed Act & Ordinance”, are hereby repealed.",
     {"The Bolan Medical College (Board of Governors) Act, 1990": "whole"}, ("balochistan", "Balochistan")),
    ("The Khyber Pakhtunkhwa Finance Ordinance, 1982", "kp", "Khyber Pakhtunkhwa",
     "(6) Section 3 of the North-West Frontier Province Finance Ordinance, 1978 (Khyber Pakhtunkhwa Ord. XII of 1978), is hereby repealed. 4.",
     {"The Khyber Pakhtunkhwa Finance Ordinance, 1978": "part"}, ("kp", "Khyber Pakhtunkhwa")),
    ("The West Pakistan Usurious Loans Ordinance, 1959", "pakistancode", "West Pakistan",
     "6. Repeal.— The enactments specified in the Schedule are hereby repealed to the extent mentioned in the fourth column thereof. "
     "SCHEDULE Year. No. Short title Extent of Repeal 1 2 3 4 1918 X The Usurious Loans Act, 1918 The whole. 1934 VII The Punjab Relief of "
     "Indebtedness Act, Sections 5 and 6. 1934",
     {"The Usurious Loans Act, 1918": "whole"}, ("pakistancode", "Federal")),
    ("The Sind Co-operative Societies (West Pakistan Amendment) Ordinance, 1965", "punjab", "West Pakistan",
     "15. Substitution of section 72, Sind VII of 1925.– For section 72 of the principal Act, the following section shall be substituted, "
     "namely:- “72. Repeal and Savings (1) The following enactments are hereby repealed:- (a) The Co-operative Societies Act, 1912; and (b) the "
     "Bombay Co-operative Societies Act, 1925, as applicable to the District of Karachi.",
     {"The Co-operative Societies Act, 1912": None}, ("pakistancode", "Federal")),
    ("The Pakistan Civil Aviation Authority Act, 2023", "pakistancode", "Federal",
     "237. Repeal.- On the commencement of this Act and the Pakistan Civil Aviation Act, 2023, the Civil Aviation Ordinance, 1960 (XXXII of 1960) "
     "and the Pakistan Civil Aviation Authority Ordinance, 1982 (XXX of 1982) shall be repealed.",
     {"The Pakistan Civil Aviation Act, 2023": None, "The Civil Aviation Ordinance, 1960": "whole",
      "The Pakistan Civil Aviation Authority Ordinance, 1982": "whole"}, ("pakistancode", "Federal")),
    ("The Sample Act, 2001", "pakistancode", "Federal",
     "5. In the Specimen Act, 1990 (V of 1990), sections 3 and 4 are hereby repealed.",
     {"The Specimen Act, 1990": "part"}, ("pakistancode", "Federal")),
    ("The Legal Practitioners and Bar Councils Act, 1973", "pakistancode", "Federal",
     "57. Repeal.- The following laws shall stand repealed, namely:— (1) The whole of the Legal Practitioners Act, 1879 (XVIII of 1879), "
     "except sections 1, 3 and 36. (2) The whole of the Legal Practitioners (Women) Act, 1923 (XXIII of 1923). (3) The whole of the Bar Councils Act, 1926",
     {"The Legal Practitioners Act, 1879": "part", "The Legal Practitioners (Women) Act, 1923": "whole"}, ("pakistancode", "Federal")),
    ("The Defence Services Act, 1972", "pakistancode", "Federal",
     "in relation to land which, when the Defence of Pakistan Ordinance, 1971 (XXX of 1971) stands repealed, is subject to any requisition",
     {"The Defence of Pakistan Ordinance, 1971": None}, ("pakistancode", "Federal")),
    ("The Competition (Exemption) Regulations, 2016", "pakistancode", "Federal",
     "12. Repeal—On coming into force of these Regulations, the Competition (Exemption) Regulations, 2014 shall stand repealed.",
     {"The Competition (Exemption) Regulations, 2014": "whole"}, ("pakistancode", "Federal")),
    ("The Sindh Public Service Commission Act, 2022", "sindh", "Sindh",
     "14. (1) The Sindh Public Service Commission Act, 1989, is hereby Repeal and saving. repealed. (2) Notwithstanding the repeal",
     {"The Sindh Public Service Commission Act, 1989": "whole"}, ("sindh", "Sindh")),
    ("The Sindh Agricultural Income Tax Act, 2018", "sindh", "Sindh",
     "23. (1) The Sindh Agricultural Income Tax Ordinance, 2000 shall Repeal and stand repealed with effect from the date this Act comes into Saving. force.",
     {"The Sindh Agricultural Income Tax Ordinance, 2000": "whole"}, ("sindh", "Sindh")),
    ("The Karachi Indecent Advertisements Act, 1960", "pakistancode", "Federal",
     "14, -Repeals.—The Karachi Indecent Advertisement Prohibition Act, 1951 (XXXVI of 1951), and the Punjab Suppression of Indecent Adver- "
     "tisements Act, 1941 (Punjab Act VII of 1941), are hereby repealed.",
     {"The Karachi Indecent Advertisement Prohibition Act, 1951": "whole"}, ("pakistancode", "Federal")),
    ("The Judges Order, 2008", "pakistancode", "Federal",
     "5. Repeal.— The salary of Judges of Superior Courts Order, 2007 (P.O.No.1 of 2007), is hereby repealed.",
     {"The Salary of Judges of Superior Courts Order, 2007": "whole"}, ("pakistancode", "Federal")),
    ("The Sample Gazette Act, 1990", "pakistancode", "Federal",
     "Clauses (f) and (g) of the University of Karachi Act, 1972 are hereby repealed. The words “in writing” are repealed wherever the "
     "Transfer of Property Act, 1882 (IV of 1882) is in force.",
     {"The University of Karachi Act, 1972": "part", "The Transfer of Property Act, 1882": None}, ("pakistancode", "Federal")),
    ("The Employees' Old-Age Benefits Act, 1976", "pakistancode", "West Pakistan",
     "82. Repeal.– The Employees’ Social Insurance Ordinance, 1962 (XXII of 1962), in its application to the Province of West Pakistan, is hereby repealed.",
     {"The Employees' Social Insurance Ordinance, 1962": "whole"}, ("pakistancode", "Federal")),
    ("The Weights and Measures (International System) Act, 1967", "pakistancode", "Federal",
     "42. Repeal.__ (1) The enactments specified in Part I of the Second Schedule shall stand repealed in an area in which or in relation to any goods "
     "or undertaking in respect of which this Act comes into force. THE SECOND SCHEDULE (See section 42) PART I 1. The Measures of Length Act, 1889 "
     "(II of 1889). 2. The Standards of Weight Act, 1939 (IX of 1939)",
     {"The Measures of Length Act, 1889": "conditional"}, ("pakistancode", "Federal")),
]

# wrong readings found when the first full run's changes were checked by hand (2026-10-18)
GAMBLING_GAZETTE = ("(2) The following enactments are hereby repealed, namely:- (a) The Baluchistan Prevention of Gambling Ordinance, 1961 "
                    "(W.P. Ordinance No. I of 1961). " + "The Gazette of Pakistan, Extraordinary, published by authority, Islamabad, "
                    "the following Act of Majlis-e-Shoora received the assent of the President and is hereby published for general "
                    "information. An Act to provide for the control of prices and the prevention of profiteering and hoarding. " * 2
                    + "The Price Control and Prevention of Profiteering and Hoarding Act, 1977")
CASES += [
    ("The Land Acquisition (West Pakistan Amendment) Ordinance, 1969", "punjab", "West Pakistan",
     "4. Repeal and savings.– (1) The following enactments and the amendments made thereby in the Land Acquisition Act, 1894 (Act I of 1894), "
     "hereinafter referred to as the said Act, are hereby repealed: (i) the Land Acquisition (N.W.F.P. Amendment) Act, 1949",
     {"The Land Acquisition Act, 1894": "part"}, ("punjab", "Federal")),
    ("The Customs Rules, 2001", "drs", "Federal",
     "Repeal.- The rules made under the following notifications are hereby repealed:- (a) S.R.O. 102 issued under section 219 of the Customs Act, 1969;",
     {"The Customs Act, 1969": None}, ("pakistancode", "Federal")),
    ("The Dowry and Bridal Gifts (Restriction) Act, 1976", "drs", "Federal",
     "Repeal.-The West Pakistan Dowry (Prohibition on Display) Act, 1967 (West Pakistan Act No. XVI of 1967) and the North-West Frontier "
     "Province Dowry Act, 1972 (N.W.F.P. Act No. IX of 1972), are hereby repealed.",
     {"The Diplomatic and Consular Privileges Act, 1972": None}, ("pakistancode", "Federal")),
    ("The Punjab Land Revenue (Abolition) Act, 1998", "punjab", "Punjab",
     "(3) It shall come into force at once and shall be deemed to have taken effect on and from the date the Punjab Land Revenue (Abolition) "
     "Ordinance 1997 (XXXI of 1997) stands repealed under Article 128 of the Constitution of the Islamic Republic of Pakistan.",
     {"The Punjab Land Revenue (Abolition) Ordinance, 1997": None}, ("punjab", "Punjab")),
    ("The Repealing Act, 1938", "drs", "Federal",
     "3. The enactments specified in the Schedule are hereby repealed to the extent mentioned in the fourth column thereof. THE SCHEDULE "
     "Year No. Short title Extent of repeal 1883 XX The Punjab District Boards Act, 1883 The whole of Chapter V. 1884 IV The Sample Act, 1884 The whole.",
     {"The Punjab District Boards Act, 1883": "part", "The Sample Act, 1884": "whole"}, ("drs", "Federal")),
    ("The Deputy Chairman and Deputy Speaker (Salaries, Allowances and Privileges) Act, 1975", "drs", "Federal",
     "Repeal.- The Chairman and Deputy Chairman of Senate (Salaries, Allowances and Privileges) Order, 1973 (P.O. No. 17 of 1973), in so far as "
     "it relates to the Chairman, and the Chairman and Speaker (Salaries, Allowances and Privileges) Ordinance, 1975 (IX of 1975) are hereby repealed.",
     {"The Chairman and Deputy Chairman of Senate (Salaries, Allowances and Privileges) Order, 1973": "part",
      "The Chairman and Speaker (Salaries, Allowances and Privileges) Ordinance, 1975": "whole"}, ("drs", "Federal")),
    ("The Validation of Laws Act, 1975", "drs", "Federal",
     "1. Short title.—(1) This Act may be called the Malaria Eradication Board (Repeal) Act, 1975. (2) Tt shall come into force at once. "
     "2. ete—(1) The Malaria Eradication Board Ordinance, 1961 CXVIII of 1961), hereinafter referred to as the Ordinance, is hereby repealed.",
     {"The Malaria Eradication Board (Repeal) Act, 1975": None, "The Malaria Eradication Board Ordinance, 1961": "whole"}, ("drs", "Federal")),
    ("The Prevention of Gambling Act, 1977", "drs", "Federal", GAMBLING_GAZETTE,
     {"The Price Control and Prevention of Profiteering and Hoarding Act, 1977": None}, ("drs", "Federal")),
    ("The Punjab Prohibition of Private Money Lending Act, 2007", "punjab", "Punjab",
     "15. Repeal.– The Punjab Money-Lenders Ordinance 1960 (W.P. Ordinance XXIV of 1960) is hereby repealed.",
     {"The West Pakistan Money-lenders Ordinance, 1960": "whole"}, ("punjab", "West Pakistan")),
    ("The Khyber Pakhtunkhwa Tibb and Homeopathic Employees (Appointment) Act, 2016", "kp", "Khyber Pakhtunkhwa",
     "6. Repeal and saving.---(1) The Khyber Pakhtunkhwa Tibb and Homeopathic employees (Regulation of Services) Act, 2014 (Khyber Pakhtunkhwa "
     "Act No. VII of 2014) is hereby repealed. (2) Notwithstanding",
     {"The Khyber Pakhtunkhwa Tibb and Homeopathic Employees (Regularization of Services) Act, 2014": "whole"}, ("kp", "Khyber Pakhtunkhwa")),
    ("The Khyber Pakhtunkhwa Probation and Parole Act, 2021", "kp", "Khyber Pakhtunkhwa",
     "Repeal.--- The Probation of Offenders Ordinance, 1960 (Ord. No. XLV of 1960), the Good Conduct Prisoners Probational Release Act, 1926 "
     "(Act No. X of 1926), are hereby repealed to the extent of the Province of Khyber Pakhtunkhwa.",
     {"The Good Conduct Prisoner's Probational Release Act, 1926": "whole"}, ("kp", "Federal")),
    ("The Balochistan Kachhi Abadis Regularization and Development of Slum Settlements Act, 2022", "balochistan", "Balochistan",
     "Repeal. (1) The Balochistan Kachhi Abadis (Regularization and Development) (Act No. II of 1987), is hereby repealed.",
     {"The Balochistan Kachi Abadies (Regularisation and Development) Act, 1987": "whole"}, ("balochistan", "Balochistan")),
    ("The Punjab Local Option (Repeal) Ordinance, 1985", "punjab", "Punjab",
     "2. Repeal of Punjab Act V of 1923.– The Punjab Local Option Act, 1923 is hereby repealed.",
     {"The Boilers Act, 1923": None}, ("punjab", "Federal")),
    ("Esta-Code", "pakistancode", "Federal",
     "The Pakistan Special Police Establishment Ordinance, 1948 (VIII of 1948), and the Special Police and Provincial Police (Amalgamation) "
     "Order, 1962 (P. O. No. I of 1962), hereinafter referred to respectively as the said Ordinance and the said Order, are hereby repealed.",
     {"The Special Police and Provincial Police (Amalgamation) Order, 1962": None}, ("drs", "Federal")),
]
CASES += [
    ("The Pakistan Civil Aviation Act, 2023", "drs", "Federal",
     '(x) "Federal Investigation Agency" means the Federal Investigation ;g"*;;;lt$"d unaei ttte Federal Investigation Agencv Act, 1974 '
     "(VIII of 1975), (y) 'ICAO' means International Civil Aviation Organization. Sample text of the definitions continues here at "
     "some length so that the repeal section begins well after them, for the time being in fori. 152. Repeal of Ordinance X)O I of 1960 and "
     "Ordinance XXX of 1962.--On the coming into force of this Act and the pakistan Airports Authority Act 2023, the Civil Aviation Ordinance, "
     "1960 (XXXI of 1960) and the pakistan Civil Aviation Ordinance, 1982 (XXX of 1982) shall be repealed.",
     {"The Federal Investigation Agency Act, 1974": None, "The Civil Aviation Ordinance, 1960": "whole"}, ("pakistancode", "Federal")),
    ("The Prevention of Gambling Act, 1977", "drs", "Federal",
     "(2) The following enactments are hereby repealed, namely:- (a) The Baluchistan Prevention of Gambling Ordinance, 1961 (W. P. Ordinance I of "
     "1961); (d) The Sind Prevention of Gambling Ordinance, 1961 (W. P. Ordinance I of 1961). ACT No. XXIX OF 1977 An Act to provide for price "
     "control and prevention of profiteering and hoarding WHEREAS it is expedient; It is hereby enacted as follows:- 1. Short title.—(1) This "
     "Act may be called the Price Control and Prevention of Profiteering and Hoarding Act, 1977.",
     {"The Price Control and Prevention of Profiteering and Hoarding Act, 1977": None}, ("pakistancode", "Federal")),
]
CASES += [
    ("The Balochistan Finance Act, 1965", "balochistan", "Balochistan",
     "(2) Section 14 of the 2[Balochistan] Finance Act, 19633, is hereby repealed. (8) Section 10 of the 10[Balochistan] Finance, Act, 196411, is hereby repealed.",
     {"The Balochistan Finance Act, 1963": "part"}, ("balochistan", "Balochistan")),
    ("The Punjab Finance Act, 1974", "punjab", "Punjab",
     "5. Repeal of section 8 of the West Pakistan6 Finance Ordinance, 1969 (W.P. Ordinance No. VII of 1969).– In the West Pakistan7 Finance "
     "Ordinance, 1969 (West Pakistan Ordinance No. VII of 1969), section 8 is repealed.",
     {"The West Pakistan Finance Ordinance, 1969": "part"}, ("punjab", "West Pakistan")),
    ("The Offence of Zina (Enforcement of Hudood) Ordinance, 1979", "drs", "Federal",
     "19. Amendment of Act XLV of 1860.—In the Pakistan Penal Code (Act XLV of 1860),-- (a) section 366, section 372, section 373, section 375 "
     "and section 376 of Chapter XVI and section 493, section 497 and section 498 of Chapter XX shall stand repealed.",
     {"The Pakistan Penal Code, 1860": "part"}, ("pakistancode", "Federal")),
]
CASES += [
    ("The State Immunity Ordinance, 1981", "pakistancode", "Federal",
     "20. Repeal .— Sections 86 and 87 of the Code of Civil Procedure, 1908 (Act V of 1908), are hereby repealed.",
     {"The Code of Civil Procedure, 1908": "part"}, ("pakistancode", "Federal")),
    ("The Code of Civil Procedure (Punjab Amendment) Act, 2020", "punjab", "Punjab",
     "3. Repeal.– The Code of Civil Procedure (Punjab Amendment),Ordinance 2020 (VIII of 2020) is hereby repealed.",
     {"The Code of Civil Procedure, 1908": None}, ("punjab", "Federal")),
    ("The Alternative Dispute Resolution Act, 2017", "pakistancode", "Federal",
     "21. Repeal.— Subject to the provisions of sub-section (3) of section 20, Section 89A of the Code of Civil Procedure, 1908 (Act V of 1908), "
     "to the extent of area or areas to which this Act is extended, shall stand repealed.",
     {"The Code of Civil Procedure, 1908": "part"}, ("pakistancode", "Federal")),
    ("Act No. XII of 1865", "drs", "Federal",
     "Act No. XVII of 1860 (to amend the Penal Code, and to authorize the transfer of prisoners) and Act No. III of 1864 are hereby repealed.",
     {"The Pakistan Penal Code": None}, ("pakistancode", "Federal")),
]
CASES += [
    # "the provisions of" the Act, with nothing between: the whole Act (for the province, here)
    ("The Sindh Trusts Act, 2020", "sindh", "Sindh",
     "120. Repeal and Savings.(1) The provisions of the Trust Act 1882 (Act No. 2 of 1882), hereafter referred to as the repealed Act, are "
     "hereby repealed to the extent of the Province of Sindh. (2)Notwithstanding the repeal under",
     {"The Trusts Act, 1882": "territory"}, ("pakistancode", "Federal")),
    ("The Punjab Example Act, 2021", "punjab", "Punjab",
     "40. Repeal.- The provisions of section 4 of the Punjab Sample Act, 1990 (V of 1990) are hereby repealed.",
     {"The Punjab Sample Act, 1990": "part"}, ("punjab", "Punjab")),
    ("The Punjab Example Act, 2022", "punjab", "Punjab",
     "40. Repeal.- The provisions of the Punjab Sample Act, 1990 (V of 1990), in so far as they relate to the licensing of dealers, are hereby repealed.",
     {"The Punjab Sample Act, 1990": "part"}, ("punjab", "Punjab")),
    # "the provisions of" the Ordinance, but only where and as notified, and "to that extent"
    ("The Punjab Weights and Measures (International System) Enforcement Act, 1975", "punjab", "Punjab",
     "32. Repeal.– On the coming into force of this Act in any area or regarding different classes of goods or classes of under-takings or "
     "establishments or commodities etc. as may be notified by Government the provisions of West Pakistan Weights and Measures Ordinance, 1965 "
     "shall stand repealed to that extent.",
     {"The West Pakistan Weights and Measures Ordinance, 1965": "part"}, ("punjab", "West Pakistan")),
    ("The University of Agriculture Faisalabad Act, 1973", "punjab", "Punjab",
     "40. Repeal and validation.– (1) The Punjab Ordinances Temporary Enactment Act, 1973, in so far as it enacts the provisions of the "
     "University of Agriculture, Faisalabad Ordinance, 1973 (Punjab Ordinance No. II of 1973), is hereby repealed.",
     {"The Punjab Ordinances Temporary Enactment Act, 1973": "part"}, ("punjab", "Punjab")),
    # the clause names the rules with one word more than the code does ("Service"): still them (read as probable)
    ("The Balochistan Civil Service (Executive Branch) Service Rules, 2025", "balochistan", "Balochistan",
     "14. Delegation. — The Government my delegate all or any of its powers under these rules to any officer subordinate to it. "
     "15. Repeal. — The Balochistan Civil Service (Executive Branch) Service Rules, 1984 are hereby repealed. BY ORDER OF GOVERNOR",
     {"The Balochistan Civil Service (Executive Branch) Rules, 1984": "whole"}, ("balochistan", "Balochistan")),
    # ... but never a word that makes it another law
    ("The Punjab Sample Service Rules, 2025", "punjab", "Punjab",
     "15. Repeal.— The Punjab Sample Service (Amendment) Rules, 1990 are hereby repealed.",
     {"The Punjab Sample Service Rules, 1990": None}, ("punjab", "Punjab")),
]
YEARS = {"The Pakistan Penal Code": 1860}
KINDS = {"Esta-Code": "Other", "The Customs Rules, 2001": "Rules"}
# law numbers the corpus records carry, for the cases matched by number
NUMBERS = {"The Balochistan Kachi Abadies (Regularisation and Development) Act, 1987": "II of 1987",
           "The Khyber Pakhtunkhwa Tibb and Homeopathic Employees (Regularization of Services) Act, 2014": "VII of 2014",
           "The Boilers Act, 1923": "V of 1923", "The Diplomatic and Consular Privileges Act, 1972": "IX of 1972",
           "The Pakistan Penal Code, 1860": "XLV of 1860"}

bad = 0
for repealer_title, source, jurisdiction, text, expected, (target_source, target_jurisdiction) in CASES:
    repealer = law(repealer_title, source, jurisdiction, kind=KINDS.get(repealer_title), text=text,
                   is_amendment="amendment" in repealer_title.lower())
    targets = {title: law(title, target_source, target_jurisdiction, number=NUMBERS.get(title), year=YEARS.get(title)) for title in expected}
    graph = kg.Graph([repealer, *targets.values()], f.same_title, f.norm_title, f.title_year)
    edges = graph.repeal_edges(repealer)
    for title, want in expected.items():
        got = [e["extent"] for e in edges if e["to"] is targets[title]]
        got = got[0] if got else None
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} want {str(want):11s} got {str(got):11s} | {title[:70]}")

# R10: a federal law repealed for the Islamabad Capital Territory and for every province, one by one, is repealed;
# for four of the five, still in force (with the four in its exceptions)
TRUST_REPEALS = [
    ("The Islamabad Capital Territory Trust Act, 2020", "pakistancode", "Federal",
     "111. Repeal.— The Trust Act, 1882 (II of 1882) is hereby repealed to the extent of Islamabad Capital Territory, Islamabad. 112. Saving."),
    ("The Punjab Trusts Act, 2020", "punjab", "Punjab", "112. Repeal and saving.− (1) The Trusts Act, 1882 (II of 1882) is hereby repealed. (2) Notwithstanding"),
    ("The Khyber Pakhtunkhwa Trust Act, 2020", "kp", "Khyber Pakhtunkhwa",
     "118. Repeal and savings.---(1) The Trust Act, 1882 (Act No. II of 1882) is hereby repealed to the extent of Khyber Pakhtunkhwa. (2) Notwithstanding"),
    ("The Balochistan Trust Act, 2020", "balochistan", "Balochistan",
     "112. Repeal. The Trust Act, 1882 is hereby repealed to the extent of Balochistan Province. 113. Power to make rules"),
    ("The Sindh Trusts Act, 2020", "sindh", "Sindh",
     "120. Repeal and Savings.(1) The provisions of the Trust Act 1882 (Act No. 2 of 1882), hereafter referred to as the repealed Act, are "
     "hereby repealed to the extent of the Province of Sindh. (2)Notwithstanding"),
]
for repealers, want in ((TRUST_REPEALS, "repealed"), (TRUST_REPEALS[:4], "in_force")):
    target = law("The Trusts Act, 1882")
    records = [law(t, s, j, text=x) for t, s, j, x in repealers] + [target]
    graph = kg.Graph(records, f.same_title, f.norm_title, f.title_year)
    for r in records:
        graph.edges += graph.repeal_edges(r)
    kg.apply_rules(graph)
    ok = target["status"] == want
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} want {want:11s} got {target['status']:11s} | R10, {len(repealers)} territories: {target['status_basis'][0][:90]}")
print("failures:", bad)
sys.exit(1 if bad else 0)
