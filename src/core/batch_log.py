
# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Logging for batch processing

import os
import time


class Log(object):
    """ Used for logging info during batch processing.
        Stores most recent analysis for each species, to stay in sync w/ data files.
        Arguments:
        1. path to log file
        2. species
        3. list of other settings of the current analysis

        LOG FORMAT, for each analysis:
        #freetext line
        species
        settings line
        files, multiple lines
    """

    def __init__(self, path, species, settings):
        # in order to append, the previous log must:
        # 1. exist
        # 2. be writeable
        # 3. match current analysis
        # On init, we parse the existing log to see if appending is possible.
        # Actual append/create happens later.
        self.possibleAppend = False
        self.filepath = path
        # self.file will be an IO stram opened by the main launcher
        self.species = species
        # Convert settings dict to formatted string
        if isinstance(settings, dict):
            parts = []
            for key, value in settings.items():
                if value and value != "None" and value != False:
                    parts.append(f"{key}={value}")
            self.settings = ', '.join(parts) if parts else 'default'
        else:
            # Legacy support for list format
            self.settings = ','.join(map(str, settings))
        self.oldAnalyses = []
        self.filesDone = []
        self.currentHeader = ""
        allans = []

        # now, check if the specified log can be resumed:
        if os.path.isfile(path):
            try:
                f = open(path, 'r+')
                print("Found log file at %s" % path)

                lines = [line.rstrip('\n') for line in f]
                f.close()
                lstart = 0
                lend = 1
                # parse to separate each analysis into
                # [freetext, species, settings, [files]]
                # (basically I'm parsing txt into json because I'm dumb)
                while lend<len(lines):
                    #print(lines[lend])
                    if len(lines[lend]) > 0:
                        if lines[lend][0] == "#":
                            allans.append([lines[lstart], lines[lstart+1], lines[lstart+2],
                                            lines[lstart+3 : lend]])
                            lstart = lend
                    lend += 1
                allans.append([lines[lstart], lines[lstart+1], lines[lstart+2],
                                lines[lstart+3 : lend]])

                # parse the log thusly:
                # if current species analysis found, store parameters
                # and compare to check if it can be resumed.
                # store all other analyses for re-printing.
                for a in allans:
                    #print(a)
                    if a[1]==self.species:
                        print("Resumable analysis found")
                        # do not reprint this in log
                        if a[2]==self.settings:
                            self.currentHeader = a[0]
                            # (a1 and a2 match species & settings anyway)
                            self.filesDone = a[3]
                            self.possibleAppend = True
                    else:
                        # store this for re-printing to log
                        self.oldAnalyses.append(a)

            except IOError:
                # bad error: lacking permissions?
                print("ERROR: could not open log at %s" % path)

    def appendFile(self, filename):
        print('Appending %s to log' % filename)
        # convert to path relative to the log file directory
        if os.path.isabs(filename):
            filename = os.path.relpath(filename, os.path.dirname(self.filepath))

        # attach file path to end of log
        self.file.write(filename)
        self.file.write("\n")
        self.file.flush()

    def getDoneFiles(self, possiblefiles):
        """ Selects files that are stored in this log from possiblefiles.
            Assumes possiblefiles stores absolute paths. """
        currdir = os.path.dirname(self.filepath)
        done_abs = [os.path.normpath(os.path.join(currdir, f)) for f in self.filesDone if not os.path.isabs(f)]
        # assuming relative paths on both lists:
        out = set(done_abs).intersection(set(possiblefiles))
        return(out)

    def appendHeader(self, header, species, settings):
        if header is None:
            header = "#Analysis started on " + time.strftime("%Y %m %d, %H:%M:%S") + ":"
        self.file.write(header)
        self.file.write("\n")
        self.file.write(species)
        self.file.write("\n")
        if type(settings) is list:
            settings = ','.join(settings)
        self.file.write(settings)
        self.file.write("\n")
        self.file.flush()

    def reprintOld(self):
        # push everything from oldAnalyses to log
        # To be called once starting a new log is confirmed
        for a in self.oldAnalyses:
            self.appendHeader(a[0], a[1], a[2])
            for f in a[3]:
                self.appendFile(f)
