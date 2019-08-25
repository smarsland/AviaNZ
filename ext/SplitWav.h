#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <locale.h>

typedef unsigned __int32 uint32_t;
typedef unsigned __int16 uint16_t;



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
} WavHeader;

typedef struct WavHeader2 {
  uint32_t Subchunk2ID;
  uint32_t Subchunk2Size;
} WavHeader2;

// Win compatibility
int strptime(char *s, char *format, struct tm *temp);
int split(char *infilearg, char *outfilearg, int t, int hasDt);

