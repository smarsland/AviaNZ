import os
import AviaNZ_batch
configdir = os.path.expanduser("~/.avianz/")
avianzbatch = AviaNZ_batch.AviaNZ_batchProcess(parent=None, mode="export", configdir=configdir)
#avianzbatch.dirName='/Users/marslast/Temp/Bats/'
avianzbatch.dirName='/Volumes/Pureora_HD1/lighting_pureora/Bat folder'
avianzbatch.outputBatPasses(avianzbatch.dirName)
