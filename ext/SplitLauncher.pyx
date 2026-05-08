cdef extern from "SplitWav.h":
		int split(char *infilearg, char *outfilearg, int t, int hasDt)
		
def launchCython(infile_c, outfile_c, cutLen, wavHasDt):
		succ = split(infile_c, outfile_c, cutLen, wavHasDt)
		return(succ)


