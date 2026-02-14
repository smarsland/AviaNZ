cdef extern from "SplitWav.h":
		int split_wav(char *infilearg, char *outfilearg, int t, int hasDt)
		
def launchCython(infile_c, outfile_c, cutLen, wavHasDt):
		succ = split_wav(infile_c, outfile_c, cutLen, wavHasDt)
		return(succ)


