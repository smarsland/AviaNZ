#!/bin/bash
set -e

# Script for mixing signal and noise examples and different SNRs.

# Need this input:
# 3 bg levels (windy at low - mid - hi SNR)
# probs ~ 5 windy ex files needed

# signal handheld - clean soundscape - noisy soundscape
# probs ~ 5 of each class
# but in all cases 5 min duration to get "purer" examples

# -----------------------------------------------------------

ROOTDIR=~/Documents/kiwis/wind/denoising
AVIANZDIR=$(dirname "$0")
cd ${ROOTDIR}
# Output dirs:
mkdir -p mixed/
mkdir -p mixed/handheld/
mkdir -p mixed/cleanss/
mkdir -p mixed/richss/

# These dirs will contain the input files:
HANDHELD=signal/handheld/
CLEANSS=signal/cleanss/
NOISYSS=signal/richss/
NOISEDIR=noise/

# 0. Download xeno-canto examples to the sources/ directory:
HHSOURCES=sources/
# XC101551
# XC121079
# XC30184
# XC409363
# XC492916
# XC561864

# # 1. Convert xeno-canto examples to appropriate format wavs:
# # (also changes names to avoid spaces!)
# ffmpeg -i "${HHSOURCES}XC101551 - Marsh Warbler - Acrocephalus palustris.mp3" \
# 	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC101551.wav
# ffmpeg -i "${HHSOURCES}XC121079 - Blackish-headed Spinetail - Synallaxis tithys.mp3" \
# 	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC121079.wav
# ffmpeg -i "${HHSOURCES}XC30184 - Common Nightingale - Luscinia megarhynchos.mp3" \
# 	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC30184.wav
# ffmpeg -i "${HHSOURCES}XC409363 - Blue-capped Ifrit - Ifrita kowaldi.mp3" \
# 	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC409363.wav
# ffmpeg -i "${HHSOURCES}XC492916 - Great Tit - Parus major newtoni.mp3" \
# 	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC492916.wav
# ffmpeg -i "${HHSOURCES}XC561864 - Wood Thrush - Hylocichla mustelina.mp3" \
# 	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC561864.wav
# 
# 
# # 2. normalize file rms levels to -28 dB
# # (ffmpeg volume is measured relative to full scale)
# cd ${HANDHELD}
# for infile in $(ls XC*.wav)
# do
# 	# normalize
# 	ffmpeg-normalize $infile -o n${infile} -nt rms -t -28 -v
# 	# move the original away
# 	mv $infile ${ROOTDIR}/${HHSOURCES}/${infile}
# done
# cd ${ROOTDIR}


# 2. define mixers. each function will take files $1, $2 and output mix into $3
mix_snr1 () {
	echo "mixing at snr 1"  # 4:1 snr rms = 16:1 snr power = 12 dB
	ffmpeg -i $1 -i $2 \
		-filter_complex \
		"[0:a]volume=0.2[a0]; \
		[1:a]volume=0.8[a1]; \
		[a0][a1]amix=inputs=2:duration=shortest,volume=2" \
		$3
}
mix_snr2 () {
	echo "mixing at snr 2"  # 1:1 snr rms = 0dB
	ffmpeg -i $1 -i $2 \
		-filter_complex \
		"[0:a]volume=0.5[a0]; \
		[1:a]volume=0.5[a1]; \
		[a0][a1]amix=inputs=2:duration=shortest,volume=2" \
		$3
}
mix_snr3 () {
	echo "mixing at snr 3"  # 1:4 snr rms = -12 dB
	ffmpeg -i $1 -i $2 \
		-filter_complex \
		"[0:a]volume=0.8[a0]; \
		[1:a]volume=0.2[a1]; \
		[a0][a1]amix=inputs=2:duration=shortest,volume=2" \
		$3
}

# other useful ffmpeg commands:
# ffmpeg -i STEREO.WAV -map_channel 0.0.0 MONO.WAV
# sox -v 0.5 LOUD.WAV QUIET.WAV
# ffmpeg -i SR.WAV -ar 16000 SR_16000.WAV

# 3. Actually mix:
SIGDIR=${HANDHELD}
si=1
echo "Mixing signals from dir ${SIGDIR}"
for SIGFILE in $(ls ${SIGDIR}/*wav)
do
	echo "Using signal example ${SIGFILE}"
	ni=1
	for NOISEFILE in $(ls ${NOISEDIR}/*wav)
	do
		echo "Using noise example ${NOISEFILE}"
		# SNR 1,2,3 are just indices for easier parsing - the actual SNR
		# is set inside the mix* functions
		mix_snr1 "${NOISEFILE}" "${SIGFILE}" "mixed/handheld/snr1_noise${ni}_sig${si}.wav"
		mix_snr2 "${NOISEFILE}" "${SIGFILE}" "mixed/handheld/snr2_noise${ni}_sig${si}.wav"
		mix_snr3 "${NOISEFILE}" "${SIGFILE}" "mixed/handheld/snr3_noise${ni}_sig${si}.wav"
		ni=$((ni+1))
	done
	si=$((si+1))
done

exit

# TODO repeat above loop with
SIGDIR=${CLEANSS}
SIGDIR=${NOISYSS}

# Batch denoise the files using AviaNZ,
# with three different noise level choices
mkdir -p denoised_const/
mkdir -p denoised_ols/
mkdir -p denoised_qr/

cd ${AVIANZDIR}
for METHOD in "const" "ols" "qr"
do
	python3 AviaNZ.py -c -n "const" \
		-d "${ROOTDIR}/mixed/handheld/" \
		-e "${ROOTDIR}/denoised_${METHOD}/handheld/"
done
