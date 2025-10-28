
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

# Configuration and filter file loading

import os
import json
from src.core import message_popup
from src.models import loader


class ConfigLoader:
    """ This deals with reading main config files.
        Not much functionality, but lots of exception handling,
        so moved it out separately.

        Most of these functions return the contents of a corresponding JSON file.
    """

    def config(self, file):
        # At this point, the main config file should already be ensured to exist.
        # It will always be in user configdir, otherwise it would be impossible to find.
        print("Loading software settings from file %s" % file)
        try:
            f = open(file, encoding='utf-8')
            config = json.load(f)
            f.close()
            return config
        except ValueError:
            # if JSON looks corrupt, quit:
            #msg = message_popup.MessagePopup("w", "Bad config file", "ERROR: file " + file + " corrupt, delete it to restore default")
            #msg.exec()
            print("ERROR: file " + file + " corrupt, delete it to restore default")
            raise

    def filters(self, dir, bats=True):
        """ Returns a dict of filter JSONs,
            named after the corresponding file names.
            bats - include bat filters?
        """
        print("Loading call filters from folder %s" % dir)
        try:
            filters = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        except Exception:
            print("Folder %s not found, no filters loaded" % dir)
            return None

        goodfilters = dict()
        for filtfile in filters:
            if not filtfile.endswith("txt"):
                continue
            # Very primitive way to recognize bat filters
            if not bats and filtfile.endswith("Bats.txt"):
                continue
            try:
                ff = open(os.path.join(dir, filtfile))
                filt = json.load(ff)
                ff.close()

                # skip this filter if it looks fishy:
                if not isinstance(filt, dict) or "species" not in filt or "SampleRate" not in filt or "Filters" not in filt or len(filt["Filters"])<1:
                    raise ValueError("Filter JSON format wrong, skipping")
                # note that method may be empty for backwards compatibility:
                if "method" in filt and (filt["method"] not in ["wv", "chp"] and filt["method"] is not None):
                    print(type(filt["method"]),filt["method"])
                    raise ValueError("Filter JSON format wrong (unrecognised method), skipping")
                for subfilt in filt["Filters"]:
                    if not isinstance(subfilt, dict) or "calltype" not in subfilt or "WaveletParams" not in subfilt or "TimeRange" not in subfilt:
                        raise ValueError("Subfilter JSON format wrong, skipping")
                    if "thr" not in subfilt["WaveletParams"] or "nodes" not in subfilt["WaveletParams"] or len(subfilt["TimeRange"])<4:
                        raise ValueError("Subfilter JSON format wrong (details), skipping")

                # if filter passed checks, store it,
                # using filename (without extension) as the key
                filtfileNoExtension = filtfile.rsplit('.', 1)[0]
                goodfilters[filtfileNoExtension] = filt
            except Exception as e:
                print("Could not load filter:", filtfile, e)
        print("Loaded filters:", list(goodfilters.keys()))
        return goodfilters

    def getNNmodels(self, filters, dirnn, targetspecies):
        """ Returns a dict of target NN models
            Filters - dict of loaded filter files
            Targetspecies - list of species names to load
            """
        print("Loading NN models from folder %s" % dirnn)
        targetmodels = dict()
        for species in targetspecies:
            filt = filters[species]
            if "NN" not in filt:
                continue
            elif filt["NN"]:
                # Determine loading method based on NN_name and available files
                nn_name = filt["NN"]["NN_name"]
                pth_path = os.path.join(dirnn, nn_name + '.pth')
                json_path = os.path.join(dirnn, nn_name + '.json')
                
                try:
                    if os.path.isfile(json_path):
                        # Use JSON loading with weights
                        model = loader.loadModelFromJson(json_path)
                        if os.path.isfile(pth_path):
                            model = loader.loadWeights(model, pth_path)
                        if 'fRange' in filt["NN"]:
                            targetmodels[nn_name] = [model, filt["NN"]["win"], filt["NN"]["inputdim"],
                                                        filt["NN"]["output"],
                                                        filt["NN"]["windowInc"], filt["NN"]["thr"], True,
                                                        filt["NN"]["fRange"]]
                        else:
                            targetmodels[nn_name] = [model, filt["NN"]["win"], filt["NN"]["inputdim"],
                                                        filt["NN"]["output"], filt["NN"]["windowInc"],
                                                        filt["NN"]["thr"], False]
                    elif os.path.isfile(pth_path):
                        # Fallback: try direct loading
                        print(f"No JSON found for {nn_name}, trying direct loading...")
                        model = loader.loadModelFromH5(pth_path)
                        # Store with species key for backward compatibility
                        targetmodels[species] = [model, filt["NN"]["win"], filt["NN"]["inputdim"], filt["NN"]["output"], filt["NN"]["windowInc"], filt["NN"]["thr"]]
                    else:
                        print(f"Warning: No model files found for {nn_name}")
                        continue
                except Exception as e:
                    print(f"Error loading model {nn_name}: {e}")
                    continue
        print("Loaded NN models:", list(targetmodels.keys()))
        return targetmodels

    def shortbl(self, file, configdir):
        # A fallback shortlist will be confirmed to exist in configdir.
        # This list is necessary
        print("Loading short species list from file %s" % file)
        try:
            if os.path.isabs(file):
                # user-picked files will have absolute paths
                shortblfile = file
            else:
                # initial file will have relative path,
                # to allow looking it up in various OSes.
                shortblfile = os.path.join(configdir, file)
            if not os.path.isfile(shortblfile):
                print("Warning: file %s not found, falling back to default" % shortblfile)
                shortblfile = os.path.join(configdir, "ListCommonBirds.txt")

            try:
                json_file = open(shortblfile, encoding='utf-8')
                readlist = json.load(json_file)
                json_file.close()
                if len(readlist)>29:
                    print("Warning: short species list has %s entries, truncating to 30" % len(readlist))
                    readlist = readlist[:29]
                return readlist
            except ValueError as e:
                # if JSON looks corrupt, quit and suggest deleting:
                print(e)
                msg = message_popup.MessagePopup("w", "Bad species list", "ERROR: file " + shortblfile + " corrupt, delete it to restore default. Reverting to default.")
                msg.exec()
                return None

        except Exception as e:
            # if file is not found at all, quit, user must recreate the file or change path
            print(e)
            msg = message_popup.MessagePopup("w", "Bad species list", "ERROR: Failed to load short species list from " + file + ". Reverting to default.")
            msg.exec()
            return None

    def longbl(self, file, configdir):
        print("Loading long species list from file %s" % file)
        try:
            if os.path.isabs(file):
                # user-picked files will have absolute paths
                longblfile = file
            else:
                # initial file will have relative path,
                # to allow looking it up in various OSes.
                longblfile = os.path.join(configdir, file)
            if not os.path.isfile(longblfile):
                print("Warning: file %s not found, falling back to default" % longblfile)
                longblfile = os.path.join(configdir, "ListDOCBirds.txt")

            try:
                json_file = open(longblfile, encoding='utf-8')
                readlist = json.load(json_file)
                json_file.close()
                return readlist
            except ValueError as e:
                print(e)
                msg = message_popup.MessagePopup("w", "Bad species list", "Warning: file " + longblfile + " corrupt, delete it to restore default. Reverting to default.")
                msg.exec()
                return None

        except Exception as e:
            print(e)
            msg = message_popup.MessagePopup("w", "Bad species list", "Warning: Failed to load long species list from " + file + ". Reverting to default.")
            msg.exec()
            return None
    
    def knownCalls(self, file, configdir):
        print("Loading known calls from file %s" % file)
        try:
            if os.path.isabs(file):
                # user-picked files will have absolute paths
                knowncallsfile = file
            else:
                # initial file will have relative path,
                # to allow looking it up in various OSes.
                knowncallsfile = os.path.join(configdir, file)
            if not os.path.isfile(knowncallsfile):
                print("Warning: file %s not found, falling back to default" % knowncallsfile)
                knowncallsfile = os.path.join(configdir, "ListKnownCalls.txt")
            try:
                knownCalls = {}
                with open(knowncallsfile, 'r') as f:
                    for line in f:
                        if not len(line)==0:
                            key, value = line.strip().split(' > ')
                            if not key in knownCalls:
                                knownCalls[key]=[]
                            knownCalls[key].append(value)
                return knownCalls
            except ValueError as e:
                print(e)
                msg = message_popup.MessagePopup("w", "Bad species list", "Warning: file " + knowncallsfile + " corrupt, delete it to restore default. Reverting to default.")
                msg.exec()
                return None

        except Exception as e:
            print(e)
            msg = message_popup.MessagePopup("w", "Bad species list", "Warning: Failed to load long species list from " + file + ". Reverting to default.")
            msg.exec()
            return None

    def batl(self, file, configdir):
        print("Loading bat list from file %s" % file)
        try:
            if os.path.isabs(file):
                # user-picked files will have absolute paths
                blfile = file
            else:
                # initial file will have relative path,
                # to allow looking it up in various OSes.
                blfile = os.path.join(configdir, file)
            if not os.path.isfile(blfile):
                print("Warning: file %s not found, falling back to default" % blfile)
                blfile = os.path.join(configdir, "ListBats.txt")

            try:
                json_file = open(blfile, encoding='utf-8')
                readlist = json.load(json_file)
                json_file.close()
                return readlist
            except ValueError as e:
                print(e)
                msg = message_popup.MessagePopup("w", "Bad species list", "Warning: file " + blfile + " corrupt, delete it to restore default. Reverting to default.")
                msg.exec()
                return None

        except Exception as e:
            print(e)
            msg = message_popup.MessagePopup("w", "Bad species list", "Warning: Failed to load bat list from " + file + ". Reverting to default.")
            msg.exec()
            return None

    def learningParams(self, file):
        print("Loading software settings from file %s" % file)
        try:
            configfile = open(file, encoding='utf-8')
            config = json.load(configfile)
            configfile.close()
            return config
        except ValueError:
            # if JSON looks corrupt, quit:
            msg = message_popup.MessagePopup("w", "Bad config file", "ERROR: file " + file + " corrupt, delete it to restore default")
            msg.exec()
            raise

    # Dumps the provided JSON array to the corresponding bird file.
    def blwrite(self, content, file, configdir):
        print("Updating species list in file %s" % file)
        try:
            if os.path.isabs(file):
                # user-picked files will have absolute paths
                file = file
            else:
                # initial file will have relative path,
                # to allow looking it up in various OSes.
                file = os.path.join(configdir, file)

            # no fallback in case file not found - don't want to write to random places.
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=1, ensure_ascii=False)

        except Exception as e:
            print(e)
            msg = message_popup.MessagePopup("w", "Unwriteable species list", "Warning: Failed to write species list to " + file)
            msg.exec()
    
    # Dumps the provided dictionary into the corresponding known calls file
    def knownCallsWrite(self, content, file, configdir):
        print("Updating known calls list in file %s" % file)
        try:
            if os.path.isabs(file):
                # user-picked files will have absolute paths
                file = file
            else:
                # initial file will have relative path,
                # to allow looking it up in various OSes.
                file = os.path.join(configdir, file)

            # no fallback in case file not found - don't want to write to random places.
            with open(file, 'w', encoding='utf-8') as f:
                for key, values in sorted(content.items()):  # Sort keys
                    for value in sorted(values):  # Sort values
                        f.write(f"{key} > {value}\n")

        except Exception as e:
            print(e)
            msg = message_popup.MessagePopup("w", "Unwriteable known calls list", "Warning: Failed to write known calls list to " + file)
            msg.exec()

    # Dumps the provided JSON array to the corresponding config file.
    def configwrite(self, content, file):
        print("Saving config to file %s" % file)
        try:
            # will always be an absolute path to the user configdir.
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=1, ensure_ascii=False)
        except Exception as e:
            print("Warning: could not save config file:")
            print(e)
