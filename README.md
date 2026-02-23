# data-quality-bench

How much label noise can an LLM safety classifier tolerate before it becomes unreliable? And does cleaning your annotation pipeline actually help, or does it make things worse?

This project benchmarks data quality degradation on real LLM safety data -- using ToxicChat, a dataset of real user prompts collected from an LLM demo and labeled by human annotators for toxicity. Two models are compared: Logistic Regression as the classical baseline and DeBERTa-v3-base as the modern encoder baseline. The experimental framework was validated on SST-2 with synthetic noise first. ToxicChat is the real-world problem.

---

## The problem this is solving

Every company shipping an LLM product needs a toxicity filter. That filter is trained on human-labeled examples. Human labelers miss edge cases. Automated pre-filtering introduces weak labels. Label corrections happen between dataset versions. That pipeline noise goes into training and nobody measures the cost until the classifier starts failing.

This project makes that cost measurable. Not just "noise hurts performance" -- but exactly where the threshold is, how different models respond, whether cleaning recovers it, and what the business impact looks like when it does not.

The framing is accurate for ToxicChat specifically: this is not annotator disagreement analysis -- labels are already aggregated via majority vote. What we are studying is real-world annotation pipeline noise: a mix of class imbalance, weak-label filtering, and residual labeling errors measured via human_annotation coverage and label corrections between dataset versions.

---

## Dataset

**ToxicChat (toxicchat0124)** -- 10,165 real user prompts from the Vicuna LLM demo. Labeled by 4 researchers using strict majority vote. 7.33% toxic, 2.01% jailbreaking. Human-AI collaborative annotation -- not all examples have full human review. Label corrections between versions 1123 and 0124 changed 1.28% of toxicity labels and 0.34% of jailbreaking labels. That correction delta is itself a noise signal.

License: CC-BY-NC-4.0 -- non-commercial use only.

---

## Models

**Logistic Regression with TF-IDF** -- the classical baseline. Fast, interpretable, and still used in production at companies that need explainability. If DeBERTa cannot beat this under noise, that is worth talking about.

**DeBERTa-v3-base** -- the modern encoder baseline. Released 2021, widely used in production text classification today. Stronger than DistilBERT on nuanced and short-text classification tasks, which makes it the right transformer choice for ToxicChat prompts. Uses disentangled attention and enhanced mask decoder -- meaningfully different architecture from BERT-family models.

Both models expose identical interfaces. Everything else -- optimizer, epochs, batch size, eval split -- is held constant across every experiment.

---

## Phase 1 results (SST-2 synthetic noise baseline)

The first phase validated the experimental framework on SST-2 sentiment classification with synthetic label noise. These results stay in the repo permanently as the controlled baseline. The interesting comparison: does synthetic noise injection accurately predict how models behave under real pipeline noise?

**Degradation curves -- F1 mean across 3 seeds:**

| Model      | 0% noise | 10% noise | 20% noise | 40% noise | Total drop |
| ---------- | -------- | --------- | --------- | --------- | ---------- |
| LogReg     | 0.9149   | 0.9042    | 0.8829    | 0.7172    | -0.1977    |
| DistilBERT | 0.9524   | 0.9418    | 0.9250    | 0.8340    | -0.1184    |

Note: Phase 1 used DistilBERT. Phase 2 uses DeBERTa-v3-base. The SST-2 numbers are kept as-is for reproducibility -- they validate the framework, not the model choice.

DistilBERT degrades smoothly. LogReg collapses nonlinearly between 20% and 40%. At 40% noise, DistilBERT retained 83.4% F1 vs LogReg's 71.7%. The exact tipping point is being located in the fine-grained sweep.

**Cleaning recovery (LogReg):**

| Strategy          | 10% noise | 20% noise | 40% noise |
| ----------------- | --------- | --------- | --------- |
| Noisy baseline    | 0.9042    | 0.8829    | 0.7172    |
| Loss filter       | 0.8886    | 0.8838    | 0.7475    |
| Heuristic filter  | 0.8983    | 0.8755    | 0.7101    |
| Confidence filter | 0.7884    | 0.7536    | 0.7419    |

Loss filter is the only strategy with consistent recovery. Confidence filter actively hurts and shows dangerous variance at high noise.

**Quantity vs quality:**

| Scenario               | LogReg F1 | DistilBERT F1 |
| ---------------------- | --------- | ------------- |
| 50k samples, 30% noise | 0.8190    | 0.8974        |
| 20k samples, clean     | 0.8823    | 0.9364        |
| Delta                  | +0.0633   | +0.0390       |

20,000 clean samples outperformed 50,000 noisy ones. Consistent across all 3 seeds.

---

## Phase 2 -- ToxicChat experiments (in progress)

**Three experiments, same structure as Phase 1:**

**Experiment 1 -- Degradation curves.** Class-conditional label flipping at increasing noise levels. Toxic labels flipped independently from non-toxic to preserve class prevalence. Primary metric: PR-AUC. Secondary metric: macro F1. PR-AUC is the standard for heavily imbalanced classification and is what ToxicChat's own paper reports.

**Experiment 2 -- Cleaning recovery.** Loss filtering with a per-class guardrail -- the filter cannot drop the minority toxic class below a minimum sample count. Tests whether cleaning strategies that work on balanced data still work when toxic examples are rare.

**Experiment 3 -- Quantity vs quality.** All data including auto-filtered weak labels vs human_annotation=True only -- smaller but higher quality. This is a real quality vs quantity comparison grounded in how the dataset was actually collected. No synthetic construction needed.

**Tipping point experiment.** Fine-grained noise sweep at 5% intervals from 5% to 40% on LogReg to find the exact noise threshold where the classifier becomes unreliable. Reported with std across seeds. "The classifier degrades nonlinearly past X% noise (±Y%)" is the finding this experiment produces.

---

## Business translation

At 100,000 user interactions per day with a 7.3% base toxic rate, there are approximately 7,300 toxic prompts to catch. At a classifier operating point tuned for maximum F1, moving from a clean to a noisy training set shifts precision and recall in ways that increase both false negatives -- toxic prompts that slip through -- and false positives -- safe prompts that get blocked. The cost of each is different. A missed toxic prompt is a safety incident. A false block is a degraded user experience. Both have measurable business cost. The exact numbers will be computed from actual ToxicChat experiment results using precision, recall, and a fixed operating threshold -- not estimated from F1 alone.

---

## Project structure

```
data-quality-bench/
├── config.py                              -- all hyperparameters and paths
├── data/
│   └── loader.py                          -- SST-2 and ToxicChat loaders
├── noise/
│   └── injector.py                        -- class-conditional label noise
├── models/
│   ├── logreg.py                          -- TF-IDF + logistic regression
│   └── deberta.py                         -- DeBERTa-v3-base fine-tuning wrapper
├── training/
│   └── trainer.py                         -- seed control, model init
├── evaluation/
│   └── evaluator.py                       -- PR-AUC, macro F1, seed aggregation
├── cleaning/
│   └── strategies.py                      -- loss, confidence, heuristic filters
├── experiments/
│   ├── run_noise_sweep.py                 -- experiment 1: degradation curves
│   ├── run_cleaning.py                    -- experiment 2: recovery study
│   ├── run_quantity_vs_quality.py         -- experiment 3: headline experiment
│   └── find_tipping_point.py             -- exact noise threshold with std
├── scripts/
│   └── save_models.py                     -- pretrain and save for Streamlit
├── streamlit_app/
│   ├── app.py                             -- interactive demo
│   └── inference.py                       -- load saved model, predict
├── notebooks/
│   └── plots.ipynb                        -- all visualizations
└── results/                               -- JSON results and PNG plots
```

---

## Stack

Python 3.10, PyTorch, HuggingFace Transformers and Datasets, scikit-learn, pandas, matplotlib, seaborn, Streamlit, sentencepiece, protobuf
