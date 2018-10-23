
python make_zooniverse.py -s 'Kiwi, Nth Is Brown' -i 'Sound Files/Batch/' -o TestOut/KNIB
python AviaNZ.py -c -s -f TestOut -o TestOut

# On laptop, have to do conda remove ffmpeg first
for f in ls TestOut/*.wav; do fn="${f%%.*}"; ffmpeg -i $fn.wav $fn.mp3; done
for f in ls TestOut/*.png; do convert $f -trim $f ; done

