# Model testing

Run the four experiments you asked for with:

```bash
cd /home/giotto/Desktop/AviaNZ/model_testing
bash run_four_experiments.sh
```

That script runs:

1. Kaytoo on its own
2. BirdNET on its own
3. RegNet + BgSub on DOC data
4. RegNet + BgSub on combined DOC + AviaNZ data

If you want to compute validation-based thresholds for the combined model after training, run:

```bash
cd /home/giotto/Desktop/AviaNZ/model_testing
python3 scripts/compute_combined_thresholds.py regnet_combined_bgsubtract_seed0
```

If you want to apply those thresholds to a prediction CSV:

```bash
cd /home/giotto/Desktop/AviaNZ/model_testing
python3 scripts/compute_combined_thresholds.py \
  regnet_combined_bgsubtract_seed0 \
  --apply-to regnet_combined_bgsubtract_seed0/predictions_somefile.csv \
  --apply-out regnet_combined_bgsubtract_seed0/predictions_somefile_thresholded.csv
```
