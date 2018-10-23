#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define __USE_XOPEN
#include <time.h>
#include <locale.h>
#include "SplitWav.h"


// This script splits a long .wav file into smaller chunks.
// output: outfile.wav_0, outfile.wav_1, ...

// COMPILATION: gcc SplitWav.c -o SplitWav
// USAGE: ./SplitWav infile.wav outfile.wav maxSeconds

int main(int argc, char *argv[]){
        // parse arguments
        FILE *infile, *outfile;
        char outfilestem[strlen(argv[2])], outfilename[strlen(argv[2])+5];

        if(argc != 4){
                fprintf(stderr, "ERROR: script needs 3 arguments, %d supplied.\n", argc-1);
                exit(1);
        }
        infile = fopen(argv[1], "rb");
        if(infile == NULL){
                fprintf(stderr, "ERROR: couldn't open input file\n");
                exit(1);
        }

        int t = atoi(argv[3]);
        if(t<1 || t>36000){
                fprintf(stderr, "ERROR: time must be between 1 s and 10 h\n");
                exit(1);
        }

        // read header in two parts:
        // up to metadata and after
        WavHeader header;
        fread(&header, sizeof(WavHeader), 1, infile);

        WavHeader2 header2;
        fread(&header2, sizeof(WavHeader2), 1, infile);

        printf("Read %u MB of data, %u channels sampled at %u Hz, %d bit depth\n", header.ChunkSize/1024/1024, header.NumChannels, header.SampleRate, header.BitsPerSample);

        // SM2 recorders produce a wamd metadata chunk:
        if(header2.Subchunk2ID==1684889975){ // wamd as uint32
                printf("-- metadata found, skipping %d bytes --\n", header2.Subchunk2Size);
                fseek(infile, header2.Subchunk2Size, SEEK_CUR);
                fread(&header2, sizeof(WavHeader2), 1, infile);
                printf("Subchunk2ID %u\n", header2.Subchunk2ID); // should be 1635017060 for data
                printf("Data size: %u bytes\n", header2.Subchunk2Size);
        }

        // init new headers for output files
        WavHeader headerN;
        headerN = header;
        WavHeader2 headerN2;
        headerN2 = header2;
        
        int numsecs = header2.Subchunk2Size / header.ByteRate + (header2.Subchunk2Size % header.ByteRate !=0);
        int numfiles = numsecs / t + (numsecs % t !=0);
        printf("%d s of input will be split into %d files of %d s\n", numsecs, numfiles, t);

        // read in part of file
        int BUFSIZE = header.ByteRate; // 1 second
        printf("-- reading chunks of %d bytes --\n", BUFSIZE);
        char linebuf[BUFSIZE];

        // parse file name
        strcpy(outfilestem, argv[2]);
        outfilestem[strlen(outfilestem)-4] = '\0';

        // parse time stamp
        int timestamp;
        struct tm timestruc;
        char timestr[17];
        setlocale(LC_ALL,"/QSYS.LIB/EN_US.LOCALE");
        if (strptime(outfilestem+strlen(outfilestem)-15, "%Y%m%d_%H%M%S", &timestruc) == NULL) {
                printf("no timestamp detected\n");
                timestamp = 0;
        } else {
                printf("timestamp detected\n");
                timestamp = 1;
                outfilestem[strlen(outfilestem)-15] = '\0';
                (&timestruc)->tm_isdst = 0;
        }

        for(int f=0; f<numfiles; f++){
                // if filename had time, change it
                // otherwise name output _0.wav etc
                if (timestamp==0){
                        sprintf(outfilename, "%s_%d.wav", outfilestem, f);
                } else {
                        mktime(&timestruc);
                        strftime(timestr, 17, "%Y%m%d_%H%M%S", &timestruc);
                        sprintf(outfilename, "%s%s.wav", outfilestem, timestr);
                        (&timestruc)->tm_sec += t;
                }
                printf("sending output to file %d/%d, %s\n", f+1, numfiles, outfilename);
                outfile = fopen(outfilename, "wb");
                if (outfile == NULL){
                        fprintf(stderr, "ERROR: couldn't open output file\n");
                        exit(1);
                }

                // file opened, so read and copy second-by-second:
                // TO CHANGE: 5-8 ChunkSize
                // 41-44 Subchunk2Size
                int secstowrite = (header2.Subchunk2Size/BUFSIZE < t*(f+1) ? header2.Subchunk2Size/BUFSIZE : t*(f+1) );
                headerN2.Subchunk2Size = (secstowrite-t*f) * header.ByteRate;
                headerN.ChunkSize = 36 + headerN2.Subchunk2Size;
                fwrite(&headerN, sizeof(WavHeader), 1, outfile);
                fwrite(&headerN2, sizeof(WavHeader2), 1, outfile);

                for(int i=t*f; i<secstowrite; i++){
                        fread(linebuf, BUFSIZE, 1, infile);
                        fwrite(linebuf, BUFSIZE, 1, outfile);
                }

                // final remainder of <1 s:
                if(header2.Subchunk2Size/BUFSIZE < t*(f+1)){
                        fread(linebuf, header2.Subchunk2Size % BUFSIZE, 1, infile);
                        printf("read last %d bytes\n", header2.Subchunk2Size % BUFSIZE);
                        fwrite(linebuf, header2.Subchunk2Size % BUFSIZE, 1, outfile);
                }

                fclose(outfile);
        }

        fclose(infile);
        
        return(0);
}
