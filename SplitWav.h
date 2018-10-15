#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct WavHeader {
  uint32_t ChunkID;
  uint32_t ChunkSize;
  uint32_t Format;
  uint32_t Subchunk1ID;
  uint32_t Subchunk1Size;
  uint16_t AudioFormat;
  uint16_t NumChannels;
  uint32_t SampleRate;
  uint32_t ByteRate;
  uint16_t BlockAlign;
  uint16_t BitsPerSample;
  uint32_t Subchunk2ID;
  uint32_t Subchunk2Size;
} WavHeader;
