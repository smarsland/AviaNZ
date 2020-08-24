#define _XOPEN_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <locale.h>
#include "SplitWav.h"


// This script splits a long .wav file into smaller chunks.
// output: outfile.wav_0, outfile.wav_1, ...

// COMPILATION: gcc SplitWav.c -o SplitWav
// USAGE: ./SplitWav infile.wav outfile.wav maxSeconds

// to deal with windows requirements for .pyds
void PyInit_SplitWav() {}

// I HATE WINDOWS
// this is handwritten strptime(s, "%Y%m%d_%H%M%S", &timestruc)

int strptime2(char *s, char *format, struct tm *temp){
	char *ptr;
	
	temp->tm_isdst = 0;
	// hopefully these don't matter
	// temp->tm_wday = 0;
	// temp->tm_yday = 0;
	
	char substr2[3];
	// s[strlen(s)-1] is \0
	strncpy(substr2, &s[strlen(s)-2], 2);
	substr2[2] = '\0';
	temp->tm_sec = strtol(substr2, &ptr, 10);
	strncpy(substr2, &s[strlen(s)-4], 2);
	substr2[2] = '\0';
	temp->tm_min = strtol(substr2, &ptr, 10);
	strncpy(substr2, &s[strlen(s)-6], 2);
	substr2[2] = '\0';
	temp->tm_hour = strtol(substr2, &ptr, 10);
	
	// then skip _
	
	strncpy(substr2, &s[strlen(s)-9], 2);
	substr2[2] = '\0';
	temp->tm_mday = strtol(substr2, &ptr, 10);
	strncpy(substr2, &s[strlen(s)-11], 2);
	substr2[2] = '\0';
	temp->tm_mon = strtol(substr2, &ptr, 10)-1;

	
	// YEAR needs 4 symbols
	char substr4[5];
	strncpy(substr4, &s[strlen(s)-15], 4);
	substr4[4] = '\0';
	temp->tm_year = strtol(substr4, &ptr, 10)-1900;
	
	return(0);
}


int split(char *infilearg, char *outfilearg, int t, int hasDt){
        // parse arguments
        FILE *infile, *outfile;
		
        // for you non-win people, just use this instead:
		// char outfilestem[strlen(outfilearg)], outfilename[strlen(outfilearg)+5];
		char *outfilestem = malloc(sizeof(char) * strlen(outfilearg));
		char *outfilename = malloc(sizeof(char) * (strlen(outfilearg)+5));
		if(outfilestem==NULL || outfilename==NULL){
			fprintf(stderr, "ERROR: could not allocate memory for arguments\n");
			exit(1);
		}

        /*if(argc != 4){
                fprintf(stderr, "ERROR: script needs 3 arguments, %d supplied.\n", argc-1);
                exit(1);
        }*/
        infile = fopen(infilearg, "rb");
        if(infile == NULL){
                fprintf(stderr, "ERROR: couldn't open input file\n");
                exit(1);
        }

        // int t = atoi(cutlen);
        if(t<1 || t>36000){
                fprintf(stderr, "ERROR: time must be between 1 s and 10 h\n");
                exit(1);
        }

        // read header in two parts:
        // up to metadata and after
        WavHeader header;
        fread(&header, sizeof(WavHeader), 1, infile);

        printf("Read %u MB of data, %u channels sampled at %u Hz, %d bit depth\n", header.ChunkSize/1024/1024, header.NumChannels, header.SampleRate, header.BitsPerSample);
        // RIFF chunk
        if(header.ChunkSize<1000 || header.ChunkID!=1179011410){
                fprintf(stderr, "ERROR: file empty or header malformed\n");
                exit(1);
        }
        if(header.Subchunk1Size>16){
                printf("%d extra format bytes found, skipping\n", header.Subchunk1Size-16);
                fseek(infile, header.Subchunk1Size-16, SEEK_CUR);
                header.Subchunk1Size = 16;
        }

        WavHeader2 header2;
        fread(&header2, sizeof(WavHeader2), 1, infile);

        int csafecount = 0;
        while (header2.Subchunk2ID!=1635017060){
                // SM2 recorders produce a wamd metadata chunk:
                printf("subchunk id: %x\n", header2.Subchunk2ID);
                if(header2.Subchunk2ID==1684889975){ // wamd as uint32
                       printf("-- metadata found, skipping %d bytes --\n", header2.Subchunk2Size);
                // _junk_ chunks for gibbon and other recordings:
                } else {
                        printf("-- unexpected chunk found, skipping %d bytes --\n", header2.Subchunk2Size);
                }
                fseek(infile, header2.Subchunk2Size, SEEK_CUR);
                fread(&header2, sizeof(WavHeader2), 1, infile);
                csafecount++;
                if (csafecount>20){
                        fprintf(stderr, "ERROR: unexpectedly many chunks found, probably misaligned WAV\n");
                        exit(1);
                }
        }
        printf("Subchunk2ID %u\n", header2.Subchunk2ID); // should be 1635017060 for data
        printf("Data size: %u bytes\n", header2.Subchunk2Size);

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
		
        // for Win compatibility:
        // char linebuf[BUFSIZE];
	char *linebuf = malloc(sizeof(char) * BUFSIZE);
	if(linebuf==NULL){
		fprintf(stderr, "ERROR: could not allocate memory for reading\n");
		exit(1);
	}
		
        // parse file name		
        strncpy(outfilestem, outfilearg, strlen(outfilearg)-4);
	outfilestem[strlen(outfilearg)-4] = '\0';

        // parse time stamp

        int timestamp;
        struct tm timestruc;
        char timestr[17];

	setlocale(LC_ALL,"/QSYS.LIB/EN_NZ.LOCALE");
        printf("%s\n", infilearg);
		
        // wish I had unix
	// if (strptime(outfilestem+strlen(outfilestem)-15, "%Y%m%d_%H%M%S", &timestruc) == NULL) {
	if (hasDt==0){
                printf("no timestamp detected\n");
                timestamp = 0;
        } else {
                printf("timestamp detected\n");
                timestamp = 1;
                strptime2(outfilestem+strlen(outfilestem)-15, "%Y%m%d_%H%M%S", &timestruc);
                outfilestem[strlen(outfilestem)-15] = '\0';

                // dealing wiht DST: just enforce original hour
                // i.e. recorders don't update their clocks,
                // so just make sure the output time is continuous.
                int orighour = (&timestruc)->tm_hour;
                mktime(&timestruc);
                if ((&timestruc)->tm_hour != orighour){
                        printf("adjusting hour to %d for DST\n", orighour);
                        (&timestruc)->tm_hour = orighour;
                }
        }

        int f;
        for(f=0; f<numfiles; f++){
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

                int i;
                for(i=t*f; i<secstowrite; i++){
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
		free(linebuf);
		free(outfilestem);
		free(outfilename);
        
        return(0);
}
