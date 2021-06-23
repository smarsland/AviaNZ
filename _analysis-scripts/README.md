## Replicating the DCASE data analysis

1. Clone this repository.

2. Ensure that you have the prerequisites for running AviaNZ (see https://github.com/smarsland/AviaNZ) and the DCASE 2018 evaluation (see https://github.com/DCASE-REPO/dcase2018_baseline).

3. Review the `main-dcase.sh` script. You may wish to set a different directory to use, and may choose to use our annotations or not.

4. Run the `main-dcase.sh` script.

5. If you chose not to use our annotations (i.e. retrain filters), you will be prompted at a stage where GUI operation is needed. See the script and accompanying paper for details on this step. After the GUI operations are finished, run the `main-dcase.sh` script again and it should proceed to completion.


## Replicating the survey data analysis

1. Clone this repository.

2. Ensure that you have the prerequisites for running AviaNZ (see https://github.com/smarsland/AviaNZ) and the DCASE 2018 evaluation (see https://github.com/DCASE-REPO/dcase2018_baseline).

3. Download audio data and annotations from the permanent repository given in the accompanying paper.

4. You may choose to redo the annotations and filter training. Follow standard AviaNZ procedures to do this via the GUI (see the software manual and the accompanying paper).

5. Review the scripts `analyze-bittern.R`, `analyze-kiwi.R` and `analyze-f1.R`. You may wish to set different processing directories.

6. Run the three scripts to replicate the analyses of interest (each script is independent of others).

7. Run `overview-and-map.R` if you wish to recreate the figures. Note that additional map images from LINZ will be needed as indicated in the script.
