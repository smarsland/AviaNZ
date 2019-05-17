# Extract 10s .wav segments
python make_zooniverse.py -s 'Kiwi (Nth Is Brown)' -i 'Sound Files/Batch2/' -o TestOut/KNIB
# Generate the spectrogram images (shallow search)
python AviaNZ.py -c -z -f TestOut -o TestOut

# On laptop, have to do conda remove ffmpeg first
# .wav to .mp3
for f in ls TestOut/*.wav; do fn="${f%%.*}"; ffmpeg -i $fn.wav $fn.mp3; done
# [nirosha@civic test]$ for f in *.wav ; do fn="/run/media/nirosha/Nirosha_Acoustics/Tier1-2015-16_Zooniverse/test/"$f; echo $fn; ffmpeg -i $fn $fn.mp3; done
# Alternative (still creates bad start and end)
# for i in Test/*.wav; do lame -b 320 -h "${i}" "${i%.wav}.mp3"; done
for f in ls TestOut/*.png; do convert $f -trim $f ; done
# [nirosha@civic test]$ for f in *.png; do convert $f -trim $f ; done

# This line copies all the data files in their folders and puts them elsewhere
# rsync -avm --include='*.data' -f 'hide,! */' . TestOut

# Rename file names in a directory in bash
# for f in *.mp3; do mv "$f" "$(echo "$f" | sed s/.wav_/_/)"; done
# Remove the fist character of filenames
# for f in *.mp3; do mv $f ${f:1}; done