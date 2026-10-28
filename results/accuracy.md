# whichmodel v0: measured accuracy

Data: 600 answers to 100 prompts from 6 local models in 3 families. All splits are by prompt (5 folds): the test prompts were never seen in training.

| Test | Accuracy | n |
|---|---|---|
| Majority-class baseline | 33.3% | 600 |
| **Full model, held-out prompts** | **66.8%** | 600 |
| Style only, held-out prompts | 66.7% | 600 |
| Words only, held-out prompts | 60.5% | 600 |
| **Unseen model** (leave-one-model-out, held-out prompts) | **31.8%** | 600 |
| Exact model (6-way, held-out prompts; baseline 16.7%) | 49.0% | 600 |

## Per family (full model, held-out prompts)

| True \ Guessed | gemma | mistral | qwen | Recall |
|---|---|---|---|---|
| gemma | 145 | 6 | 49 | 72.5% |
| mistral | 18 | 141 | 41 | 70.5% |
| qwen | 51 | 34 | 115 | 57.5% |

## Unseen model: can it place a model it never trained on?

| Held-out model | True family | Accuracy | What it guessed |
|---|---|---|---|
| gemma-4-12b | gemma | 30.0% | qwen 63, gemma 30, mistral 7 |
| gemma-4-e4b-it-q4-k-m | gemma | 46.0% | qwen 48, gemma 46, mistral 6 |
| hearthfire-24b-q4-k-s | mistral | 19.0% | qwen 55, gemma 26, mistral 19 |
| ministral-3-3b-instruct-2512-q4-k-m | mistral | 24.0% | qwen 59, mistral 24, gemma 17 |
| qwen3.6-35b-a3b-ud-q6-k | qwen | 35.0% | gemma 35, qwen 35, mistral 30 |
| qwen3.8-27b-ud-iq4-xs | qwen | 37.0% | qwen 37, gemma 34, mistral 29 |

## Length: accuracy when the test text is cut to its first N words

| Words | Accuracy |
|---|---|
| full (median 248) | 66.8% |
| 200 | 67.2% |
| 100 | 63.3% |
| 50 | 56.5% |
| 25 | 51.8% |

## Calibration: does "80%" mean right 80% of the time?

Out-of-fold predictions. The shipped model divides logits by T = 3.75, fitted on these same out-of-fold predictions (it changes confidence, never the top guess).

| Top-guess confidence | n | Mean confidence | Actual accuracy |
|---|---|---|---|
| raw 0%-50% | 13 | 46% | 31% |
| raw 50%-70% | 59 | 60% | 36% |
| raw 70%-90% | 100 | 80% | 54% |
| raw 90%-100% | 428 | 98% | 75% |
| calibrated 0%-50% | 93 | 44% | 39% |
| calibrated 50%-70% | 229 | 60% | 59% |
| calibrated 70%-90% | 202 | 80% | 81% |
| calibrated 90%-100% | 76 | 95% | 88% |

## Strongest fingerprints per family (final model weights)

- **gemma**: hr lines, uses the word "breakdown", signature phrase "intricate", straight apostrophe, uses the word "depending", open ah okay, signature phrase "tapestry", uses the word "comprehensive"
- **mistral**: open great, em dash, uses the word "here's", parenthesis, signature phrase "here s", uses the word "some", uses the word "you're", uses the word "needed"
- **qwen**: curly quotes, uses the word "core", uses the word "not", signature phrase "fantastic", uses the word "off", signature phrase "important to note", uses the word "great", signature phrase "testament"
