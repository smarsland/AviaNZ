#!/bin/bash
set -e

# Script for mixing signal and noise examples and different SNRs.

# Need this input:
# windy background examples in denoising/noise/
# clean soundscapes in denoising/signal/richss/
# xeno-canto examples in denoising/sources/ , self-downloaded

# -----------------------------------------------------------

# Set these two as needed:
ROOTDIR=~/Documents/kiwis/wind/deposited/denoising  # path containing input wavs. will also store output
AVIANZDIR=~/Documents/gitrep/birdscape/  # path containing AviaNZ.py
cd ${ROOTDIR}

# These dirs will contain the input files:
# (relative to ROOTDIR which is current working dir)
HANDHELD=signal/handheld/
NOISYSS=signal/richss/
NOISEDIR=noise/

# Output dirs:
mkdir -p mixed/
mkdir -p mixed/handheld/
mkdir -p mixed/richss/

# 0. Download xeno-canto examples to the sources/ directory:
HHSOURCES=sources/
# XC101551
# XC121079
# XC30184
# XC409363
# XC492916
# XC561864

# 1. Convert xeno-canto examples to appropriate format wavs:
# (also changes names to avoid spaces!)
ffmpeg -i "${HHSOURCES}XC101551 - Marsh Warbler - Acrocephalus palustris.mp3" \
	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC101551.wav
ffmpeg -i "${HHSOURCES}XC121079 - Blackish-headed Spinetail - Synallaxis tithys.mp3" \
	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC121079.wav
ffmpeg -i "${HHSOURCES}XC30184 - Common Nightingale - Luscinia megarhynchos.mp3" \
	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC30184.wav
ffmpeg -i "${HHSOURCES}XC409363 - Blue-capped Ifrit - Ifrita kowaldi.mp3" \
	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC409363.wav
ffmpeg -i "${HHSOURCES}XC492916 - Great Tit - Parus major newtoni.mp3" \
	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC492916.wav
ffmpeg -i "${HHSOURCES}XC561864 - Wood Thrush - Hylocichla mustelina.mp3" \
	-t 120 -ar 16000 -map_channel 0.0.0 ${HANDHELD}/XC561864.wav


# 2. normalize file rms levels to -28 dB
# (ffmpeg volume is measured relative to full scale)
cd ${HANDHELD}
for infile in $(ls XC*.wav)
do
	# normalize
	ffmpeg-normalize $infile -o n${infile} -nt rms -t -28 -v
	# move the original away
	mv $infile ${ROOTDIR}/${HHSOURCES}/${infile}
done
cd ${ROOTDIR}


# 3. define mixers. each function will take files $1, $2 and output mix into $3
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

# 4a. Actually mix:
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

# 4b. Repeat the mixing with the other signals
SIGDIR=${NOISYSS}
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
		mix_snr1 "${NOISEFILE}" "${SIGFILE}" "mixed/richss/snr1_noise${ni}_sig${si}.wav"
		mix_snr2 "${NOISEFILE}" "${SIGFILE}" "mixed/richss/snr2_noise${ni}_sig${si}.wav"
		# (not using snr3 b/c this it too queit for the non-targeted signal files)
		ni=$((ni+1))
	done
	si=$((si+1))
done

# 5. Batch denoise the files using AviaNZ,
# with three different noise level choices
cd ${AVIANZDIR}
for METHOD in "ols" "qr" "const" 
do
	# for handheld signals
	mkdir -p ${ROOTDIR}/denoised_${METHOD}/handheld/
	python3 AviaNZ.py -c -n ${METHOD} \
		-d "${ROOTDIR}/mixed/handheld/" \
		-e "${ROOTDIR}/denoised_${METHOD}/handheld/"
	# for soundscape signals
	mkdir -p ${ROOTDIR}/denoised_${METHOD}/richss/
	python3 AviaNZ.py -c -n ${METHOD} \
		-d "${ROOTDIR}/mixed/richss/" \
		-e "${ROOTDIR}/denoised_${METHOD}/richss/"
done
