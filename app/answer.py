"""
app/answer.py

From a question to an answer: the language model (DeepSeek, OpenAI-compatible API) is used twice --

    understand   which language the question is in (English, Urdu, Roman Urdu) and, when it is not English, its
                 English wording for the search (the laws are in English; translated questions find far more,
                 see eval/score.py)
    answer       an answer written only from the numbered sections the search returned, every statement cited
                 [n], in the language the question was asked in

The model never answers from memory: if the sections do not answer the question it says so.
"""

import json
import re

from openai import OpenAI

from app.config import LLM_BASE_URL, LLM_MODEL, secret

LANGUAGES = {"en": "English", "ur": "Urdu (in Urdu script)", "ru": "Roman Urdu (Urdu written in English letters)"}
ROMAN_URDU = re.compile(r"\b(?:kya|hai|hain|ka|ki|ke|mein|main|kaise|kese|karna|kar|nahi|nahin|agar|kis|kab|kon|kaun|mujhe|mera|meri|"
                        r"apna|apni|wala|wali|liye|chahiye|sakta|sakti|hota|hoti|ho|aur|bhi|ye|yeh|woh|wo|jo|ko|se|par|talaq|nikah|"
                        r"jaidad|wirasat|kiraya|zamanat|dafa)\b", re.I)
URDU_SCRIPT = re.compile(r"[؀-ۿ]")

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        key = secret("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (setx DEEPSEEK_API_KEY \"...\")")
        _client = OpenAI(api_key=key, base_url=LLM_BASE_URL)
    return _client


def chat(messages: list[dict], json_mode: bool = False, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    response = client().chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens,
        response_format={"type": "json_object"} if json_mode else None,
        extra_body={"thinking": {"type": "disabled"}},    # the fast mode: these tasks need no long reasoning
    )
    return response.choices[0].message.content or ""


ISSUES_AT = 350        # characters: a question longer than this is split into the issues it raises
QUESTIONS_IN = re.compile(r"(?m)^\s*(?:\d{1,2}|[a-z])[.)]\s|\?\s+\S")    # numbered parts, or another question after a "?"
MAX_ISSUES = 24        # every issue the plan lists, as a rule: a tighter cap starved one part or another

TRANSLATE = ("Give the question in plain English, keeping every legal term, law name and section number, and using the "
             "words Pakistani statutes use. Urdu words with an everyday meaning are written out by that meaning, but only where "
             "the question uses them: kirayedar is \"tenant of rented premises (a house or shop)\" and malik-makan \"landlord of "
             "rented premises\" (an agricultural tenant only when land or crops are mentioned); zamanat is bail, and zamanat "
             "qabl az giraftari pre-arrest bail; khula is dissolution of marriage at the wife's instance.")
SPLIT = ("The question asks several things. \"parts\" has one entry for each thing asked, in order: every numbered sub-question "
         "is a part, and a question without numbers may still ask several things. \"asks\" is that part in a few words. "
         "\"issues\" are the distinct legal points that part turns on, one to seven, most important first -- different points, "
         "never rewordings of one point -- each a short self-contained search query in English naming the point and the "
         "facts it turns on, in the words a Pakistani statute would use: \"director's duty to disclose his interest in a "
         "contract with a firm of his relative\", \"dishonestly issuing a cheque that is dishonoured\", \"petition by minority "
         "shareholders against oppression and mismanagement of a company\". A part about offences has an issue for each "
         "offence the facts suggest, under any law (penal, company, tax, foreign exchange, electronic crimes). A part asking "
         "several points of procedure has one issue for each, naming what it concerns: \"agency that investigates criminal "
         "breach of trust and forgery\", \"court that tries the offence of money laundering\", \"pre-arrest bail under "
         "section 498 of the Code of Criminal Procedure for an accused of criminal breach of trust\". Name a law only where "
         "the question names it or you are sure the point belongs to it, and a section number only where you are certain "
         "of it -- a section named is fetched as it stands, so a wrong number brings the wrong text.")


def issues_of(parts: list) -> list[str]:
    """The issues searched, up to MAX_ISSUES: every part's first, so no part of the question goes unsearched; then the
    rest to whichever part has most left ("what offences has he committed" raises six, "can he get bail" one)."""
    lists = [list(dict.fromkeys(i.strip() for i in p.get("issues") or [] if isinstance(i, str) and i.strip()))
             for p in parts if isinstance(p, dict)]
    lists = [issues for issues in lists if issues]
    taken = [1 if issues else 0 for issues in lists]
    while sum(taken) < MAX_ISSUES and any(t < len(issues) for t, issues in zip(taken, lists)):
        most = max(range(len(lists)), key=lambda n: (len(lists[n]) - taken[n], -n))
        taken[most] += 1
    out = [issue for issues, t in zip(lists, taken) for issue in issues[:t]]
    return list(dict.fromkeys(out))[:MAX_ISSUES]


def several_issues(question: str) -> bool:
    return len(question) > ISSUES_AT or len(QUESTIONS_IN.findall(question)) >= 2


def understand(question: str) -> dict:
    """{"language", "english", "issues"}: the language the question is in, the question in English for the search, and
    -- for a long question or one asking several things -- the separate legal issues it raises, each searched on its
    own (one search for six questions finds none of them well). The script decides the language -- Urdu letters, or
    Urdu words in Latin letters (Roman Urdu) -- since the model tends to call Roman Urdu "Urdu"."""
    if URDU_SCRIPT.search(question):
        language = "ur"
    elif len(ROMAN_URDU.findall(question)) >= 2:
        language = "ru"
    else:
        language = "en"
    split = several_issues(question)
    if language == "en" and not split:
        return {"language": language, "english": question, "issues": [], "parts": []}
    shape = '{"english": "..."' + (', "parts": [{"asks": "...", "issues": ["...", "..."]}]' if split else "") + "}"
    reply = chat([
        {"role": "system", "content": "You prepare questions about Pakistani law for a search over English statutes. Reply with JSON: "
                                      f"{shape}. {TRANSLATE}" + (f" {SPLIT}" if split else "") + " Do not answer the question."},
        {"role": "user", "content": question}], json_mode=True, max_tokens=1500 if split else 300, temperature=0)   # the same reading every time
    try:
        parsed = json.loads(reply)
        english = (parsed.get("english") or question).strip()
        parts = [p for p in parsed.get("parts") or [] if isinstance(p, dict)] if split else []
        issues = issues_of(parts)
    except (json.JSONDecodeError, AttributeError, TypeError):
        english, issues, parts = question, [], []
    return {"language": language, "english": english, "issues": issues, "parts": parts}


SYSTEM = """You are Pakistan Law Assistant, a legal research and drafting assistant for law students and members of the public in
Pakistan. You work ONLY from the numbered sources given with each question: provisions of Pakistani statutes (the Constitution,
federal and provincial Acts and Ordinances, inherited West Pakistan laws, and rules and regulations), each marked with where it
applies and its status.

THE LAW YOU RELY ON
- State only what the sources say. Do not add any rule, punishment, time limit, fee or procedure from memory, even if you believe it
  is correct, and never rely on foreign law. If the question turns on court decisions or practice that the statutes do not settle,
  say that the statutes provided do not settle it.
- Respect the hierarchy of Pakistani law: the Constitution prevails over statutes; an Act or Ordinance prevails over rules and
  regulations made under it.
- Jurisdiction matters. Federal law applies throughout Pakistan except where a source says it is repealed for a territory; a
  province's law applies only in that province; the Islamabad Capital Territory follows federal law. Since the Eighteenth
  Amendment many subjects are provincial, so when the sources show federal and provincial versions, or versions of different
  provinces, say which applies where. If the user names a province, answer for that province.
- Status. All sources given to you are law in force, except those marked STATUS NOT CONFIRMED: Sindh laws whose current status
  could not be confirmed from the Sindh Code. When you rely on one, say so plainly. Never call any law repealed or amended unless a
  source says so.
- If the sources do not answer the question, or answer only part of it, say so clearly for that part, and say what the sources
  do cover if it helps. Sources that are not about the question are simply left out: do not mention them.

HOW YOU ANSWER A QUESTION
- Begin with a direct answer in two or three sentences. Then explain, under short headings where useful: what the law provides,
  who it applies to, conditions and exceptions, procedure step by step (who acts, before which authority or court, within what
  time, in what form), and consequences or penalties -- each only as far as the sources go.
- Cite every statement of law with the number of its source in square brackets, like [2] or [1][3]. Never cite a number that is
  not in the sources. The first time you use a provision, name it in full: "section 7(1) of the Muslim Family Laws Ordinance,
  1961 [1]", "Article 199 of the Constitution [3]".
- Use the provision's own structure and words: sub-sections (1), (2); clauses (a), (b); provisos. Put the key words of the law in
  quotation marks exactly as printed. To quote a provision at length, quote it as a block (each line starting with "> "),
  word for word with its numbering, never paraphrased inside the quotation.
- Write in {language}. Keep the names of laws, section and article numbers, and legal terms of art (talaq, khula, bail, FIR,
  decree) as written in the sources, explaining a term in {language} where a student would need it.
- Be precise and neutral: no opinions on religion or politics, no moral judgement of the user.
- When the law sets a number -- a share of capital, an amount, a period, an age -- and the facts give one, state both
  and compare them in so many words before concluding ("Bilal holds 9%, which is less than the 10% required"); where
  the law counts several persons together ("member or members holding not less than ten percent"), say so.

WHEN ASKED TO DRAFT A DOCUMENT
(a notice, legal notice, application, complaint, petition, plaint, reply, affidavit, agreement, deed, power of attorney, or similar)
- Draft it in the form used in Pakistan for that document, on the provisions in the sources, and put the whole document inside one
  fenced block opening with ```document and closing with ```. Inside the block, plain text laid out as a legal document:
    * the court, authority or addressee heading in capitals where the document goes to one ("IN THE COURT OF THE SENIOR CIVIL
      JUDGE, [CITY]"; "BEFORE THE CHAIRMAN, UNION COUNCIL [NAME], [CITY]"; "To," and the addressee's particulars);
    * the title in capitals naming the law it is made under ("NOTICE OF TALAQ UNDER SECTION 7(1) OF THE MUSLIM FAMILY LAWS
      ORDINANCE, 1961"), and "Subject:" or the parties' description ("[Name] son/daughter/wife of [Name], resident of [Address]
      ... Petitioner / Applicant VERSUS [Name] ... Respondent") as the document requires;
    * for applications, petitions and plaints, "Respectfully Sheweth:" followed by numbered paragraphs (1., 2., 3.) setting out the
      facts and the grounds in law, then "PRAYER:" beginning "It is, therefore, most respectfully prayed that ...";
    * for affidavits, "I, [Name], ..., do hereby solemnly affirm and declare as under:", numbered paragraphs, and "VERIFICATION:
      Verified on oath at [Place] on this [day] day of [Month], [Year] that the contents of the above affidavit are true and correct
      to the best of my knowledge and belief and nothing has been concealed therefrom.";
    * for agreements and deeds, the parties, recitals ("WHEREAS ..."), numbered clauses, and "IN WITNESS WHEREOF ...";
    * the closing: signature lines, name, CNIC number, address, date and place, and witnesses (name, CNIC, signature) where the law
      or practice requires them.
- Use a placeholder in square brackets for every fact you have not been given -- [Name of wife], [CNIC No.], [Date],
  [Address] -- and never invent names, dates, CNIC numbers, case numbers, amounts or addresses.
- Do not put source numbers inside the document. Write the document in English, the language of courts and offices, unless the
  user asks for Urdu; then write it in Urdu.
- After the block, in {language}: "Based on:" with the provisions relied on and their source numbers [n]; the blanks the user must
  fill in; to whom, where and how the document is to be sent or filed, and any time limit, as far as the sources say; and one line
  advising that a lawyer review it before use.
- Refuse to draft anything meant to deceive: a false affidavit, a backdated document, an imitation of an official record.

LIMITS
- If a question is not about the law in Pakistan, say briefly that you can only help with Pakistani law.
- End every reply with one short line, in {language}, saying this is legal information, not legal advice."""


def source_block(n: int, chunk: dict) -> str:
    where = chunk.get("status_applies_in") or chunk.get("jurisdiction") or ""
    status = "STATUS NOT CONFIRMED" if chunk["tier"] == "unconfirmed" else "in force"
    for exception in chunk.get("status_exceptions") or []:
        status += f"; repealed for the {exception['territory']}"
    return f"[{n}] {chunk['title']} -- {chunk['label']} ({where}; {status})\n{chunk['text'].strip()}"


def write_answer(question: str, language: str, sources: list[dict], province: str | None, english: str | None = None,
                 issues: list[str] | None = None) -> str:
    if not sources:
        return ""
    context = "\n\n".join(source_block(n, chunk) for n, chunk in enumerate(sources, 1))
    place = f"\nThe student is asking about the law as it applies in: {province}." if province else ""
    # the reading the search used ("kirayedar": the tenant of a rented house or shop, not of farm land), so the
    # answer is about the same thing
    meant = f"\nIn English, the question means: {english}" if english and english.strip() != question.strip() else ""
    # a question of several parts: the sources were searched issue by issue, and the answer takes them in turn
    listed = ("\nThe sources were searched for these issues. Answer the question part by part, under a heading for each, "
              "and say plainly for any part the sources do not cover:\n" + "\n".join(f"{n}. {i}" for n, i in enumerate(issues, 1))) if issues else ""
    return chat([
        {"role": "system", "content": SYSTEM.format(language=LANGUAGES.get(language, "English"))},
        {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}{meant}{place}{listed}"}],
        max_tokens=6000 if issues else 3500)    # room for a full draft, or for every part of a long question


MAX_GAPS = 8


def find_gaps(question: str, answer: str) -> list[str]:
    """What a first answer says its sources did not cover, as search queries -- searched again, and the answer
    rewritten with what turns up. A call of its own: asked to add a line at the end of a long answer, the model
    did so only some of the time."""
    reply = chat([
        {"role": "system", "content": "You check an answer about Pakistani law against the question it answers. Reply with JSON: "
                                      '{"gaps": ["...", "..."]} -- one short English search query for each point of the question '
                                      "that the answer says its sources do not cover, do not contain or cannot settle. Name the "
                                      "legal point in the words a statute would use, and the law where you are sure of it; a "
                                      "section number only where you are certain (pre-arrest bail: section 498 of the Code of "
                                      "Criminal Procedure). Leave out questions of fact and points only case law could settle. "
                                      f"At most {MAX_GAPS}; [] if the answer covers everything."},
        {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}"}], json_mode=True, max_tokens=600, temperature=0)
    try:
        return [g.strip() for g in json.loads(reply).get("gaps") or [] if isinstance(g, str) and g.strip()][:MAX_GAPS]
    except (json.JSONDecodeError, AttributeError):
        return []


def cited_numbers(answer: str, count: int) -> list[int]:
    """The source numbers an answer cites, in order of first citation, ignoring any out of range."""
    seen = []
    for n in map(int, re.findall(r"\[(\d{1,2})\]", answer)):
        if 1 <= n <= count and n not in seen:
            seen.append(n)
    return seen
