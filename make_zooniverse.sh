
python make_zooniverse.py -s 'Kiwi (Nth Is Brown)' -i 'Sound Files/Batch2/' -o TestOut/KNIB
python AviaNZ.py -c -z -f TestOut -o TestOut

# On laptop, have to do conda remove ffmpeg first
for f in ls TestOut/*.wav; do fn="${f%%.*}"; ffmpeg -i $fn.wav $fn.mp3; done
for f in ls TestOut/*.png; do convert $f -trim $f ; done

# This line copies all the data files in their folders and puts them elsewhere
# rsync -avm --include='*.data' -f 'hide,! */' . TestOut

