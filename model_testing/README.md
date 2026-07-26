# Model testing workflow

This folder contains the end-to-end workflow for comparing:

1. Kaytoo performance on its own.
2. BirdNET performance on its own.
3. RegNet + BgSub + kbird prior 2.0 on DOC-only data.
4. RegNet + BgSub + kbird prior 2.0 on combined DOC + AviaNZ data.

## Recommended run order

### 1. Build the datasets

From the repository root:

```bash
cd model_testing
bash build_dataset.sh
bash build_combined_dataset.sh
```

These produce:

- matched data: a matched DOC/AviaNZ split
- combined data: DOC + AviaNZ data for the full combined experiment

### 2. Run the main comparison pipeline

The easiest entry point is:

```bash
cd model_testing
bash run_summary.sh
```

This will run the experiments in the order below:

- BirdNET pretrained evaluation
- RegNet baseline (matched DOC)
- RegNet + BgSub (matched DOC)
- Kaytoo pretrained evaluation
- Kaytoo finetuned
- RegNet + BgSub on full DOC
- RegNet + BgSub on combined DOC + AviaNZ
- RegNet + BgSub combined + matched finetune

### 3. Analyse the results

The script above already calls the analysis step automatically:

```bash
python3 scripts/analyze_all_results.py matched_tests --output matched_tests/analysis
python3 scripts/analyze_all_results.py full_doc_tests --output full_doc_tests/analysis
python3 scripts/analyze_all_results.py combined_tests --output combined_tests/analysis
python3 scripts/make_summary_figure.py
```

What this does:

- reads the per-experiment result files
- computes adaptive thresholds per split
- computes cross-threshold metrics (thresholds from one split applied to the other)
- writes summary CSVs and the summary figure

## What each script is for

- run_summary.sh
  - trains/evaluates the comparison models in one shot
- analyze_all_results.py
  - aggregates experiment outputs and computes the metrics used by the summary figure
- make_summary_figure.py
  - produces the publication-style summary plots
- compute_combined_thresholds.py
  - computes a per-class threshold table for a trained model

## Thresholds: what to use and why

There are two different notions of thresholding here:

1. Thresholds for the summary figure
   - analyze_all_results.py uses “oracle” thresholds tuned on the same split that is being evaluated.
   - This is useful for comparison, but it is slightly optimistic because the thresholds are chosen from the same data being scored.
   - The “cross” metrics show how well thresholds from one split transfer to the other.

2. Thresholds for real deployment / final model selection
   - use a separate validation set, not the final test set.
   - tune thresholds on validation data only.
   - then evaluate on the untouched test set once.

### Recommended practice

For the best combined RegNet model:

1. Train on the training split.
2. Tune per-class thresholds on a validation split from the combined dataset.
3. Freeze those thresholds.
4. Evaluate once on the test split.

This avoids using test data twice.

### What the current scripts do

The existing script compute_combined_thresholds.py will:

- prefer predictions_val.csv when it exists
- use the trainer’s internal validation predictions to tune thresholds
- write thresholds_combined.csv for the model directory

That is the best current proxy for the “validation-based” thresholding workflow.

### Important caveat about the matched 9-species data

The matched test data only contains 9 species. That means:

- thresholds for those 9 classes can be tuned reasonably well
- thresholds for other classes are either not covered or are only weakly estimated

Because the models are trained on a broader class vocabulary than the matched split, threshold tuning should ideally be done on a validation set that includes the full class set you care about. If you only tune on the matched 9-species subset, you are effectively tuning a restricted subset of the classifier.

## Practical recommendation

If your goal is to compare models fairly:

- use the existing run_summary.sh + analyze_all_results.py + make_summary_figure.py flow for the paper-style comparison.

If your goal is to choose the final operating thresholds:

- use the combined model directory and run:

```bash
python3 scripts/compute_combined_thresholds.py model_testing/regnet_combined_bgsubtract_seed0
```

This writes a threshold table that is appropriate for the combined model.

If you want the most principled thresholding setup, add a dedicated validation split from the combined data and use that for threshold tuning before the final test evaluation.

### Commands to run for the validation-based workflow

Train the combined model as usual, then compute thresholds from the saved validation predictions:

```bash
cd model_testing
python3 scripts/compute_combined_thresholds.py \
  regnet_combined_bgsubtract_seed0
```

If you want to apply those thresholds to a matched-test prediction CSV (for example, after evaluating on the matched test split), run:

```bash
cd model_testing
python3 scripts/compute_combined_thresholds.py \
  regnet_combined_bgsubtract_seed0 \
  --apply-to regnet_combined_bgsubtract_seed0/predictions_avianz.csv \
  --apply-out regnet_combined_bgsubtract_seed0/predictions_avianz_thresholded.csv
```

If you want to use a specific validation CSV instead of the default predictions_val.csv, pass it explicitly:

```bash
cd model_testing
python3 scripts/compute_combined_thresholds.py \
  regnet_combined_bgsubtract_seed0 \
  --prediction-csv regnet_combined_bgsubtract_seed0/predictions_val.csv
```
