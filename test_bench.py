import json, bench
# 1. The Python step runs arithmetic and refuses anything else.
assert bench.run_python("result = 254453 - 222358") == "32095"
assert bench.run_python("import math\nresult = round(math.sqrt(16), 1)") == "4.0"
assert bench.run_python("import os\nresult = os.getcwd()") == "refused"
assert bench.run_python("result = open('x').read()") == "refused"
assert bench.run_python("result = 1/0") == "error"
assert bench.run_python("") is None
# 2. Formatting matches the reference answers' style.
assert bench.fmt("32095", "millions") == "$32,095 (in millions)" and bench.fmt("0.0453", "percent") == "4.53%"
# 3. The exact numeric check: same number within 1%, percents within 0.1 points, text references skipped.
assert bench.numeric_match("$32,095 (in millions)", "$32,095 (in millions)") is True
assert bench.numeric_match("$32,095 (in millions)", "32,100") is True
assert bench.numeric_match("$32,095 (in millions)", "$29,000") is False
assert bench.numeric_match("4.53%", "4.5%") is True and bench.numeric_match("4.53%", "0.0453") is True and bench.numeric_match("4.53%", "4.8%") is False
assert bench.numeric_match("Yes", "Yes") is None
assert bench.numeric_match("Operating leases should be treated as debt because they are fixed obligations", "x") is None
# 4. Final-answer extraction.
assert bench.final_line("work...\nFinal answer: $9,285 (in millions)") == "$9,285 (in millions)"
# 5. End to end with a fake model: the code condition executes Python, the judge result is recorded, and the report renders.
rows = [{"context": "ctx", "question": "What is Gross Profit in 2024?", "answer": "$32,095 (in millions)", "question_type": "basic"},
        {"context": "ctx", "question": "What is adjusted EBITDA?", "answer": "$11,969 (in millions)", "question_type": "assumption"},
        {"context": "ctx", "question": "Why treat leases as debt?", "answer": "Because they are fixed financing obligations", "question_type": "conceptual"}]
def fake_llm(model, messages, as_json=False):
    u = messages[-1]["content"]
    if messages[0]["content"].startswith("You grade"): return json.dumps({"correct": "32,095" in u.split("Answer to grade:")[-1] or "fixed" in u.split("Answer to grade:")[-1]})
    if as_json: return json.dumps({"inputs": ["Revenue 254,453", "COGS 222,358"], "python": "result = 254453 - 222358", "unit": "millions", "text": ""}) if "Gross" in u else json.dumps({"inputs": [], "python": "", "unit": "text", "text": "Leases are fixed obligations"})
    return "Final answer: $30,000 (in millions)"
import tempfile, pathlib, os
bench.llm = fake_llm; bench.dataset = lambda: rows; os.environ["OPENAI_API_KEY"] = "test"
tmp = pathlib.Path(tempfile.mkdtemp()); bench.ROOT = tmp
bench.main(0, "fake", "fake-judge", 2, ["baseline", "code"])
raw = json.loads((tmp / "results" / "raw.json").read_text())
assert raw["fake|code|0"]["answer"] == "$32,095 (in millions)" and raw["fake|code|0"]["correct"] is True
assert raw["fake|baseline|0"]["correct"] is False and raw["fake|baseline|0"]["numeric"] is False
print("all tests passed")
# 6. Held-out split: stratified, disjoint, and examples only ever come from the dev half.
rows2 = [{"question": f"What is adjusted EBITDA variant {i}?", "question_type": ["basic", "assumption", "conceptual"][i % 3], "chain_of_thought": "x" * 3000, "answer": str(i), "context": "c"} for i in range(30)]
dev, test = bench.split(rows2)
assert not set(dev) & set(test) and len(dev) + len(test) == 30
for i in test:
    assert set(bench.pick_examples(rows2[i], rows2, dev)) <= set(dev)
instr = bench.examples_instr(rows2[test[0]], rows2, dev)
assert instr.startswith(bench.EX_HEAD) and instr.count("Example question:") == 3 and " ..." in instr
print("held-out tests passed")
