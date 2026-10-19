# financeqa-assume

Can a prompt pipeline close FinanceQA's assumption gap without fine-tuning?

[FinanceQA](https://huggingface.co/datasets/AfterQuery/FinanceQA) is AfterQuery's benchmark of 148 analyst questions on Costco's 10-K. In the [paper](https://arxiv.org/abs/2501.18062), GPT-4o answered 2.2% of the assumption questions correctly, the ones where the 10-K leaves out an input and an analyst has to fill it in. The paper closes much of that gap by fine-tuning. This tests three ways to answer every question with the same model and no training:

1. **Answer directly**, like the paper.
2. **State inputs and assumptions first.** List every input the answer needs, quoting it from the 10-K or naming the assumption an analyst would make.
3. **State inputs, compute in Python.** Same, but the arithmetic is written as code and actually executed.

## Results

<!-- results:start -->
_Run it to populate._
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
