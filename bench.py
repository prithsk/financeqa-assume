"""financeqa-assume: can a prompt pipeline close FinanceQA's assumption gap without fine-tuning?

Three ways to answer each of FinanceQA's 148 questions on Costco's 10-K, same model:
  baseline - answer from the context, like the paper
  assume   - list every input first, quoting it or stating the analyst assumption that fills it, then answer
  code     - same, but the arithmetic is written as Python and actually executed
  examples - three expert worked solutions from the other half of the benchmark in context, tested only on held-out questions
A different model judges every answer against AfterQuery's reference, and numeric answers get an exact check too.
"""
import argparse, concurrent.futures as cf, datetime as dt, json, os, pathlib, re, subprocess, sys, time, urllib.request

ROOT = pathlib.Path(__file__).parent
PAPER = {"basic": 0.447, "assumption": 0.022, "conceptual": 0.625}   # GPT-4o, Table 1 of arXiv 2501.18062
ROWS = "https://datasets-server.huggingface.co/rows?dataset=AfterQuery%2FFinanceQA&config=default&split=test&offset={o}&length=100"

def load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

def fetch_json(url):
    for i in range(5):
        try: return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "financeqa-assume"}), timeout=120).read())
        except Exception:
            if i == 4: raise
            time.sleep(3 * (i + 1))

def dataset():
    cache = ROOT / "data" / "financeqa.json"
    if cache.exists(): return json.loads(cache.read_text(encoding="utf-8"))
    rows, o = [], 0
    while True:
        page = fetch_json(ROWS.format(o=o))["rows"]; rows += [r["row"] for r in page]
        if len(page) < 100: break
        o += 100
    cache.parent.mkdir(exist_ok=True); cache.write_text(json.dumps(rows), encoding="utf-8"); return rows

_client = None
def llm(model, messages, as_json=False):
    global _client
    if _client is None:
        from openai import OpenAI; _client = OpenAI(max_retries=8)
    kw = {"response_format": {"type": "json_object"}} if as_json else {}
    r = _client.chat.completions.create(model=model, messages=messages, temperature=0, **kw)
    return r.choices[0].message.content or ""

SYSTEM = "You are a financial analyst at an investment bank. Use the 10-K excerpt provided."
def base_msgs(row, instr):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"10-K excerpt:\n{row['context']}\n\nQuestion: {row['question']}\n\n{instr}"}]

BASE_INSTR = "Answer the question. End with one line that starts with 'Final answer:'."
ASSUME_INSTR = ("Before answering, list every input the answer needs. For each input, either quote the number from the excerpt, "
                "or, if the excerpt does not give it, state the assumption a professional analyst would make and why. "
                "Then compute the answer. End with one line that starts with 'Final answer:'.")
CODE_INSTR = ("List every input the answer needs. For each input, either quote the number from the excerpt, or, if the excerpt does not "
              "give it, state the assumption a professional analyst would make and why. Then write Python that computes the answer from "
              "those inputs and stores it in a variable named result. Use only arithmetic and the math module. If the question needs no "
              "calculation, leave python empty and put the answer in text. Reply as JSON with keys: inputs (list of strings), "
              "python (string), unit (one of: millions, percent, ratio, per_share, yes_no, text), text (string).")

BLOCKED = re.compile(r"\b(import\s+(?!math\b)\w+|from\s+\w+\s+import|open|exec|eval|compile|__\w+__|subprocess|os\.|sys\.|input)\b")
def run_python(code):
    """Run model-written arithmetic in a separate, isolated Python with a timeout. Anything beyond math is refused."""
    if not code.strip(): return None
    if BLOCKED.search(code): return "refused"
    prog = code + "\nprint(repr(result))"
    try:
        out = subprocess.run([sys.executable, "-I", "-c", prog], capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else "error"
    except subprocess.TimeoutExpired: return "timeout"

def fmt(value, unit):
    try: v = float(value)
    except (TypeError, ValueError): return str(value)
    if unit == "percent": return f"{v * 100:.2f}%" if abs(v) < 1.5 else f"{v:.2f}%"
    if unit == "millions": return f"${v:,.0f} (in millions)"
    if unit == "per_share": return f"${v:,.2f}"
    return f"{v:,.4g}"

def final_line(text):
    m = re.findall(r"final answer\s*:\s*(.+)", text, flags=re.I)
    return (m[-1] if m else text.strip().splitlines()[-1] if text.strip() else "").strip()

def split(rows):
    """Stratified halves: within each question type, alternate questions go to dev and test. Test questions never appear as examples."""
    dev, test = [], []
    for t in sorted({r["question_type"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["question_type"] == t]
        dev += idx[0::2]; test += idx[1::2]
    return sorted(dev), sorted(test)

def words(s): return set(re.findall(r"[a-z]+", (s or "").lower())) - {"the", "a", "of", "for", "in", "is", "what", "and", "to", "year", "ending"}
def pick_examples(row, rows, dev, k=3):
    q = words(row["question"])
    score = lambda j: len(q & words(rows[j]["question"])) / max(1, len(q | words(rows[j]["question"]))) + (0.1 if rows[j]["question_type"] == row["question_type"] else 0)
    return sorted(dev, key=lambda j: -score(j))[:k]

EX_HEAD = "Here are worked solutions by an expert analyst to related questions on the same 10-K:"
def examples_instr(row, rows, dev):
    ex = []
    for j in pick_examples(row, rows, dev):
        cot = rows[j]["chain_of_thought"] or ""
        ex.append(f"Example question: {rows[j]['question']}\nExpert solution: {cot[:2500]}{' ...' if len(cot) > 2500 else ''}\nAnswer: {rows[j]['answer']}")
    return EX_HEAD + "\n\n" + "\n\n".join(ex) + "\n\nNow answer the question the same way an expert analyst would. " + BASE_INSTR

CTX = {}
def answer(row, cond, model):
    if cond == "examples": return final_line(llm(model, base_msgs(row, examples_instr(row, CTX["rows"], CTX["dev"]))))
    if cond == "baseline": return final_line(llm(model, base_msgs(row, BASE_INSTR)))
    if cond == "assume": return final_line(llm(model, base_msgs(row, ASSUME_INSTR)))
    raw = llm(model, base_msgs(row, CODE_INSTR), as_json=True)
    try: j = json.loads(raw)
    except json.JSONDecodeError: return final_line(raw)
    res = run_python(j.get("python", ""))
    if res in (None, "refused", "error", "timeout"): return (j.get("text") or "").strip() or (res or "")
    return fmt(res, j.get("unit", "text"))

JUDGE = ("You grade answers to financial analysis questions against a reference answer. A numeric answer is correct when it matches the "
         "reference value within 1% relative difference after putting both in the same unit and scale. A percent is correct within 0.1 "
         "percentage points. Yes or no answers must match. A text answer is correct when it reaches the same conclusion as the reference. "
         'Reply as JSON: {"correct": true or false}.')
def judge(row, ans, model):
    raw = llm(model, [{"role": "system", "content": JUDGE},
                      {"role": "user", "content": f"Question: {row['question']}\nReference answer: {row['answer']}\nAnswer to grade: {ans}"}], as_json=True)
    try: return bool(json.loads(raw).get("correct"))
    except json.JSONDecodeError: return "true" in raw.lower()

NUM = re.compile(r"-?\$?\(?-?[\d,]*\.?\d+\)?%?")
YEAR = re.compile(r"(19|20)\d\d")
def first_number(s, skip_years=True):
    """The first number in the text, skipping bare years like 2024 so "for 2024 is $9,285" reads as 9,285."""
    for m in NUM.finditer(s or ""):
        if skip_years and YEAR.fullmatch(m.group()): continue
        t = m.group().replace("$", "").replace(",", "").replace("%", "")
        neg = t.startswith("(") or t.startswith("-"); t = t.strip("()-")
        try: return (-1 if neg else 1) * float(t), m.group().endswith("%")
        except ValueError: continue
    return None
def numeric_match(gold, ans):
    """Exact check for numeric references: same number within 1%, or within 0.1 points for percents. None if not numeric."""
    g = first_number(gold, skip_years=False); a = first_number(ans)
    if g is None or re.search(r"[A-Za-z]{4,}", re.sub(r"\(in millions\)|millions?|billions?", "", gold or "", flags=re.I)): return None
    if a is None: return False
    if g[1]: return abs(g[0] - a[0]) <= 0.1 or abs(g[0] - a[0] * 100) <= 0.1
    return abs(g[0] - a[0]) <= 0.01 * max(1.0, abs(g[0]))

def main(n, model, judge_model, workers, conds):
    load_env()
    if not os.environ.get("OPENAI_API_KEY"): raise SystemExit("Put OPENAI_API_KEY in .env")
    rows = dataset(); dev, test = split(rows); CTX.update(rows=rows, dev=dev)
    held = "examples" in conds
    evals = (test if held else list(range(len(rows))))[: n or None]
    rawf = ROOT / "results" / "raw.json"; rawf.parent.mkdir(exist_ok=True)
    raw = json.loads(rawf.read_text(encoding="utf-8")) if rawf.exists() else {}
    def job(i, cond):
        key = f"{model}|{cond}|{i}"
        if key in raw: return key, raw[key]
        a = answer(rows[i], cond, model); return key, {"answer": a, "correct": judge(rows[i], a, judge_model), "numeric": numeric_match(rows[i]["answer"], a)}
    for cond in conds:
        todo = [i for i in evals if f"{model}|{cond}|{i}" not in raw]
        with cf.ThreadPoolExecutor(workers) as ex:
            for k, (key, v) in enumerate(ex.map(lambda i: job(i, cond), todo), 1):
                raw[key] = v
                if k % 10 == 0 or k == len(todo): rawf.write_text(json.dumps(raw), encoding="utf-8"); print(f"  {cond}: {k}/{len(todo)}", flush=True)
        rawf.write_text(json.dumps(raw), encoding="utf-8")
    types = ["basic", "assumption", "conceptual"]
    def acc(cond, t=None):
        idx = [i for i in evals if t is None or rows[i]["question_type"] == t]
        return sum(raw[f"{model}|{cond}|{i}"]["correct"] for i in idx) / max(1, len(idx)), len(idx)
    names = {"baseline": "Answer directly", "assume": "State inputs and assumptions first", "code": "State inputs, compute in Python",
             "examples": "Three expert worked solutions in context"}
    md = [f"# FinanceQA without fine-tuning, {dt.date.today()}", "",
          (f"Held-out half: {len(evals)} of 148 FinanceQA questions. The other half supplies the expert examples and is never tested. " if held else "") +
          f"{len(evals)} FinanceQA questions on Costco's 10-K. Answering model `{model}`, graded by `{judge_model}` against AfterQuery's reference answers.", "",
          "| Question type | Questions | Paper, GPT-4o | " + " | ".join(names[c] for c in conds) + " |", "|---|---|---|" + "---|" * len(conds)]
    for t in types:
        n_t = acc(conds[0], t)[1]
        md.append(f"| {t} | {n_t} | {PAPER[t]:.1%} | " + " | ".join((f"{acc(c, t)[0]:.1%}" if n_t else "-") for c in conds) + " |")
    md.append(f"| all | {len(evals)} | | " + " | ".join(f"**{acc(c)[0]:.1%}**" for c in conds) + " |")
    for k, v in raw.items():                                   # recompute the exact check from cached answers, so fixes apply without new calls
        i = int(k.split("|")[2])
        if k.startswith(model + "|") and i in set(evals): v["numeric"] = numeric_match(rows[i]["answer"], v["answer"])
    agree = [(v["correct"], v["numeric"]) for k, v in raw.items() if k.startswith(model + "|") and int(k.split("|")[2]) in set(evals) and v["numeric"] is not None]
    md += ["", "## Grader check", "",
           f"On the {len(agree)} graded answers with a numeric reference, the judge agreed with an exact numeric check {sum(a == b for a, b in agree) / max(1, len(agree)):.1%} of the time." if agree else "No numeric answers to cross-check.",
           "", "## Honest limits", "",
           "- One company, 148 questions, one run at temperature 0. A gap of a few questions within a type is noise, especially for the smaller types.",
           "- The paper's grading code is no longer public, so this grades with a separate judge model plus an exact numeric check. The baseline column against the paper's column is the calibration.",
           "- The prompts were written from the paper's description of the task, before looking at any reference answers, and are the same for every question.",
           "- The expert examples come only from the dev half. All 148 questions are on the same 10-K, so a held-out question can share building blocks with a dev example, like adjusted EBITDA and adjusted EBITDA margin. That is the point of expert examples, and also why this is one company, not a general result.",
           "- The Python step runs model-written arithmetic in an isolated subprocess with a 10 second timeout and refuses anything beyond the math module."]
    out = "\n".join(md); (ROOT / "results" / f"{dt.date.today()}.md").write_text(out, encoding="utf-8"); print(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="first N questions only, for a cheap smoke test")
    ap.add_argument("--model", default="gpt-4o"); ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--conds", default="baseline,assume,code")
    a = ap.parse_args(); main(a.n, a.model, a.judge, a.workers, a.conds.split(","))
