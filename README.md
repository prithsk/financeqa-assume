# financeqa-assume

Can a prompt pipeline close FinanceQA's assumption gap without fine-tuning?

[FinanceQA](https://huggingface.co/datasets/AfterQuery/FinanceQA) is AfterQuery's benchmark of 148 analyst questions on Costco's 10-K. In the [paper](https://arxiv.org/abs/2501.18062), GPT-4o answered 2.2% of the assumption questions correctly, the ones where the 10-K leaves out an input and an analyst has to fill it in. The paper closes much of that gap by fine-tuning. This tests three ways to answer every question with the same model and no training:

1. **Answer directly**, like the paper.
2. **State inputs and assumptions first.** List every input the answer needs, quoting it from the 10-K or naming the assumption an analyst would make.
3. **State inputs, compute in Python.** Same, but the arithmetic is written as code and actually executed.

## Results

<!-- results:start -->
# FinanceQA without fine-tuning, 2026-10-19

Held-out half: 74 of 148 FinanceQA questions. The other half supplies the expert examples and is never tested. 74 FinanceQA questions on Costco's 10-K. Answering model `gpt-4o`, graded by `gpt-4o-mini` against AfterQuery's reference answers.

| Question type | Questions | Paper, GPT-4o | Answer directly | State inputs and assumptions first | Three expert worked solutions in context |
|---|---|---|---|---|---|
| basic | 19 | 44.7% | 42.1% | 36.8% | 57.9% |
| assumption | 23 | 2.2% | 0.0% | 8.7% | 17.4% |
| conceptual | 32 | 62.5% | 65.6% | 53.1% | 75.0% |
| all | 74 | | **39.2%** | **35.1%** | **52.7%** |

## Grader check

On the 196 graded answers with a numeric reference, the judge agreed with an exact numeric check 93.4% of the time.

## Honest limits

- One company, 148 questions, one run at temperature 0. A gap of a few questions within a type is noise, especially for the smaller types.
- The paper's grading code is no longer public, so this grades with a separate judge model plus an exact numeric check. The baseline column against the paper's column is the calibration.
- The prompts were written from the paper's description of the task, before looking at any reference answers, and are the same for every question.
- The expert examples come only from the dev half. All 148 questions are on the same 10-K, so a held-out question can share building blocks with a dev example, like adjusted EBITDA and adjusted EBITDA margin. That is the point of expert examples, and also why this is one company, not a general result.
- The Python step runs model-written arithmetic in an isolated subprocess with a 10 second timeout and refuses anything beyond the math module.
## Calibration on all 148 questions

Answering directly, GPT-4o scored 44.7% on basic questions, the paper's exact number, and 0 of 46 on assumption questions against the paper's 1 of 46. On all 148, stating assumptions first scored 31.1% and computing in executed Python scored 24.3%, both below answering directly at 33.8%.

<!-- results:end -->

## Calibration and grading

The paper's grading code is no longer public. Every answer is graded by a different model against AfterQuery's reference answer, and every answer with a numeric reference also gets an exact check, same number within 1% or a percent within 0.1 points. The report shows how often the two agree. The direct-answer column next to the paper's own GPT-4o numbers is the calibration.

## Run it

```
cp .env.example .env            # add OPENAI_API_KEY
python -m pip install -r requirements.txt
python test_bench.py
python bench.py --n 10          # smoke test, about 50 cents
python bench.py                 # all 148 questions, three ways
```

Every answer is cached in results/raw.json, so a rerun only does what's missing.

## Safety

The Python step runs model-written arithmetic in a separate isolated Python process with a 10 second timeout. Imports beyond the math module, file access and system calls are refused before anything runs.

## License

MIT. FinanceQA is AfterQuery's, under Apache 2.0.
