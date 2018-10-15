#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "SplitWav.h"


// This script splits a long .wav file into smaller chunks.
// output: outfile.wav_0, outfile.wav_1, ...

// COMPILATION: gcc SplitWav.c -o SplitWav
// USAGE: ./SplitWav infile.wav outfile.wav maxSeconds

int main(int argc, char *argv[]){
        // parse arguments
        FILE *infile, *outfile;
        char outfilestem[strlen(argv[2])], outfilename[strlen(argv[2])];

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

        // read header
        WavHeader header;
        fread(&header, sizeof(WavHeader), 1, infile);
        WavHeader headerN;
        headerN = header;
        
        printf("Read %u MB of data, %u channels sampled at %u Hz\n", header.ChunkSize/1024/1024, header.NumChannels, header.SampleRate);

        int numsecs = header.Subchunk2Size / header.ByteRate + (header.Subchunk2Size % header.ByteRate !=0);
        int numfiles = numsecs / t + (numsecs % t !=0);
        printf("File will be split into %d files of %d seconds\n", numfiles, t);

        // read in part of file
        int BUFSIZE = header.ByteRate; // 1 second
        printf("reading chunks of %d bytes\n", BUFSIZE);
        char linebuf[BUFSIZE];
        strcpy(outfilestem, argv[2]);
        outfilestem[strlen(outfilestem)-4] = '\0';
        for(int f=0; f<numfiles; f++){
                sprintf(outfilename, "%s_%d.wav", outfilestem, f);
                printf("sending output to file %d/%d, %s\n", f+1, numfiles, outfilename);
                outfile = fopen(outfilename, "wb");
                if (outfile == NULL){
                        fprintf(stderr, "ERROR: couldn't open output file\n");
                        exit(1);
                }

                // file opened, so read and copy second-by-second:
                // TO CHANGE: 5-8 ChunkSize
                // 41-44 Subchunk2Size
                int secstowrite = (header.Subchunk2Size/BUFSIZE < t*(f+1) ? header.Subchunk2Size/BUFSIZE : t*(f+1) );
                headerN.Subchunk2Size = (secstowrite-t*f) * header.ByteRate;
                headerN.ChunkSize = 36 + headerN.Subchunk2Size;
                fwrite(&headerN, sizeof(WavHeader), 1, outfile);

                for(int i=t*f; i<secstowrite; i++){
                        fread(linebuf, BUFSIZE, 1, infile);
                        fwrite(linebuf, BUFSIZE, 1, outfile);
                }

                // final remainder of <1 s:
                if(header.Subchunk2Size % BUFSIZE > 0){
                        fread(linebuf, header.Subchunk2Size % BUFSIZE, 1, infile);
                        printf("read last %d bytes\n", header.Subchunk2Size % BUFSIZE);
                        fwrite(linebuf, header.Subchunk2Size % BUFSIZE, 1, outfile);
                }

                fclose(outfile);
        }

        fclose(infile);
        
        return(0);
}
