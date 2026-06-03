# Bird Classification — Key Experiment Summary

Eight experiments selected to tell the story of training data quality vs quantity,
augmentation, and cross-dataset generalisation.

- **AviaNZ** = Waitākere Ranges data (reliable labels, ~24 species)
- **DOC** = Department of Conservation data (noisy labels, ~130 species, 12 tested)
- **Kaytoo** = reference model trained on all DOC data (including noisy labels)
- **BirdNet** = external pretrained model (Google, not all NZ species)

> threshold 0.5 = fixed operating point; tuned = per-class thresholds optimised on each split

## Threshold = 0.5 (fixed)

| Model | AviaNZ F1 | DOC F1 | AviaNZ Acc | AviaNZ Acc (lab) | DOC Acc | DOC Acc (lab) |
| --- | --- | --- | --- | --- | --- | --- |
| BirdNet (pretrained) | 0.165 | 0.234 | 45.2% | 19.4% | 62.9% | 28.1% |
| RegNet Baseline | 0.133 | 0.448 | 50.3% | 13.5% | 69.9% | 41.9% |
| RegNet +BgSub | 0.189 | 0.472 | 49.1% | 26.5% | 71.1% | 46.1% |
| Kaytoo (pretrained) | 0.305 | 0.501 | 35.9% | 34.2% | 51.6% | 43.7% |
| Kaytoo (finetuned) | 0.045 | 0.091 | 11.4% | 17.4% | 23.4% | 30.5% |
| RegNet Scale N=8k | 0.356 | 0.448 | 39.8% | 37.4% | 41.9% | 58.7% |
| RegNet N=8k+FT | 0.217 | 0.541 | 56.3% | 15.5% | 69.6% | 52.1% |

## Per-class tuned thresholds (†)

| Model | AviaNZ F1† | DOC F1† | AviaNZ Acc† | AviaNZ Acc† (lab) | DOC Acc† | DOC Acc† (lab) |
| --- | --- | --- | --- | --- | --- | --- |
| BirdNet (pretrained) | 0.260 | 0.505 | 0.0% | 0.0% | 64.6% | 31.1% |
| RegNet Baseline | 0.298 | 0.567 | 1.5% | 3.2% | 73.7% | 43.7% |
| RegNet +BgSub | 0.358 | 0.632 | 25.4% | 18.1% | 76.1% | 53.9% |
| Kaytoo (pretrained) | 0.486 | 0.631 | 40.1% | 42.6% | 68.7% | 53.3% |
| Kaytoo (finetuned) | 0.309 | 0.397 | 39.5% | 15.5% | 44.8% | 37.1% |
| RegNet Scale N=8k | 0.467 | 0.545 | 38.0% | 52.3% | 68.2% | 57.5% |
| RegNet N=8k+FT | 0.432 | 0.641 | 45.5% | 42.6% | 72.0% | 67.7% |

---

*F1 = macro-F1 over species present in the test set.  Acc = exact-match accuracy (all files).  Acc (lab) = accuracy on labelled files only.*
