# Version 3.0 14/09/20
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti

# This is the script that starts AviaNZ. It processes command line options
# and then calls either part of the GUI, or runs on the command line directly.

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2020

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

import click
import sys

# Command line running to run a filter is something like
# python AviaNZ.py -c -b -d "/home/marslast/Projects/AviaNZ/Sound Files/train5" -r "Morepork" -w

# For training
# python AviaNZ.py -c -t -d "/home/marslast/Projects/AviaNZ/Sound Files/train5" -e "/home/marslast/Projects/AviaNZ/Sound Files/train6" -r "Morepork" -x 2

# For testing
# python AviaNZ.py -c -u -d "/home/marslast/Projects/AviaNZ/Sound Files/test1" -r "Kiwi (Tokoeka Rakiura)"
@click.command()
@click.option('-c', '--cli', is_flag=True, help='Run in command-line mode')
@click.option('-s', '--cheatsheet', is_flag=True, help='Make the cheatsheet images')
@click.option('-z', '--zooniverse', is_flag=True, help='Make the Zooniverse images and sounds')
@click.option('-f', '--infile', type=click.Path(), help='Input wav file (mandatory in CLI mode)')
@click.option('-o', '--imagefile', type=click.Path(), help='If specified, a spectrogram will be saved to this file')
@click.option('-b', '--batchmode', is_flag=True, help='Batch processing')
@click.option('-t', '--training', is_flag=True, help='Train a CNN recogniser')
@click.option('-u', '--testing', is_flag=True, help='Train a recogniser')
@click.option('-d', '--sdir1', type=click.Path(), help='Input sound directory, training or batch processing')
@click.option('-e', '--sdir2', type=click.Path(), help='Second input sound directory, training')
@click.option('-r', '--recogniser', type=str, help='Recogniser name (without ".txt"), batch processing')
@click.option('-w', '--wind', is_flag=True, help='Apply wind filter')
@click.option('-x', '--width', type=float, help='Width of windows for CNN')
@click.argument('command', nargs=-1)
def mainlauncher(cli, cheatsheet, zooniverse, infile, imagefile, batchmode, training, testing, sdir1, sdir2, recogniser, wind, width, command):
    try:
        import platform, os, json, shutil
        from jsonschema import validate
        import SupportClasses
    except Exception as e:
        print("ERROR: could not import packages")
        raise

    # determine location of config file and bird lists
    if platform.system() == 'Windows':
        # Win
        configdir = os.path.expandvars(os.path.join("%APPDATA%", "AviaNZ"))
    elif platform.system() == 'Linux' or platform.system() == 'Darwin':
        # Unix
        configdir = os.path.expanduser("~/.avianz/")
    else:
        print("ERROR: what OS is this? %s" % platform.system())
        raise

    # if config and bird files not found, copy from distributed backups.
    # so these files will always exist on load (although they could be corrupt)
    # (exceptions here not handled and should always result in crashes)
    if not os.path.isdir(configdir):
        print("Creating config dir %s" % configdir)
        try:
            os.makedirs(configdir)
        except Exception as e:
            print("ERROR: failed to make config dir")
            print(e)
            raise

    # pre-run check of config file validity
    confloader = SupportClasses.ConfigLoader()
    configschema = json.load(open("Config/config.schema"))
    try:
        config = confloader.config(os.path.join(configdir, "AviaNZconfig.txt"))
        validate(instance=config, schema=configschema)
        print("successfully validated config file")
    except Exception as e:
        print("Warning: config file failed validation with:")
        print(e)
        try:
            shutil.copy2("Config/AviaNZconfig.txt", configdir)
        except Exception as e:
            print("ERROR: failed to copy essential config files")
            print(e)
            raise

    # check and if needed copy any other necessary files
    necessaryFiles = ["ListCommonBirds.txt", "ListDOCBirds.txt", "ListBats.txt", "LearningParams.txt"]
    for f in necessaryFiles:
        if not os.path.isfile(os.path.join(configdir, f)):
            print("File %s not found in config dir, providing default" % f)
            try:
                shutil.copy2(os.path.join("Config", f), configdir)
            except Exception as e:
                print("ERROR: failed to copy essential config files")
                print(e)
                raise

    # copy over filters to ~/.avianz/Filters/:
    filterdir = os.path.join(configdir, "Filters/")
    if not os.path.isdir(filterdir):
        print("Creating filter dir %s" % filterdir)
        os.makedirs(filterdir)
    for f in os.listdir("Filters"):
        ff = os.path.join("Filters", f) # Kiwi.txt
        if not os.path.isfile(os.path.join(filterdir, f)): # ~/.avianz/Filters/Kiwi.txt
            print("Recogniser %s not found, providing default" % f)
            try:
                shutil.copy2(ff, filterdir) # cp Filters/Kiwi.txt ~/.avianz/Filters/
            except Exception as e:
                print("Warning: failed to copy recogniser %s to %s" % (ff, filterdir))
                print(e)

    # run splash screen:
    if cli:
        print("Starting AviaNZ in CLI mode")
        if batchmode:
            import AviaNZ_batch
            if os.path.isdir(sdir1) and recogniser in confloader.filters(filterdir).keys():
                avianzbatch = AviaNZ_batch.AviaNZ_batchProcess(parent=None, mode="CLI", configdir=configdir, sdir=sdir1, recogniser=recogniser, wind=wind)
                print("Analysis complete, closing AviaNZ")
            else:
                print("ERROR: valid input dir (-d) and recogniser name (-r) are essential for batch processing")
                raise
        elif training:
            import Training
            if os.path.isdir(sdir1) and os.path.isdir(sdir2) and recogniser in confloader.filters(filterdir).keys() and width>0:
                training = Training.CNNtrain(configdir,filterdir,sdir1,sdir2,recogniser,width,CLI=True)
                training.cliTrain()
                print("Training complete, closing AviaNZ")
            else:
                print("ERROR: valid input dirs (-d and -e) and recogniser name (-r) are essential for training")
                raise
        elif testing:
            import Training
            filts = confloader.filters(filterdir)
            if os.path.isdir(sdir1) and recogniser in filts:
                testing = Training.CNNtest(sdir1, filts[recogniser], recogniser, configdir,filterdir,CLI=True)
                print("Testing complete, closing AviaNZ")
            else:
                print("ERROR: valid input dir (-d) and recogniser name (-r) are essential for training")
                raise
        else:
            if (cheatsheet or zooniverse) and isinstance(infile, str):
                import AviaNZ
                avianz = AviaNZ(configdir=configdir, CLI=True, cheatsheet=cheatsheet, zooniverse=zooniverse,
                                firstFile=infile, imageFile=imagefile, command=command)
                print("Analysis complete, closing AviaNZ")
            else:
                print("ERROR: valid input file (-f) is needed")
                raise
    else:
        task = None
        print("Starting AviaNZ in GUI mode")
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        # a hack to fix default font size (Win 10 suggests 7 pt for QLabels for some reason)
        QApplication.setFont(QApplication.font("QMenu"))

        while True:
            # splash screen?
            if task is None:
                # This screen asks what you want to do, then processes the response
                import Dialogs
                first = Dialogs.StartScreen()
                first.show()
                app.exec_()
                task = first.getValues()

            avianz = None
            if task == 1:
                import AviaNZ_manual
                avianz = AviaNZ_manual.AviaNZ(configdir=configdir)
            elif task==2:
                import AviaNZ_batch_GUI
                avianz = AviaNZ_batch_GUI.AviaNZ_batchWindow(configdir=configdir)
            elif task==3:
                import AviaNZ_batch_GUI
                avianz = AviaNZ_batch_GUI.AviaNZ_reviewAll(configdir=configdir)
            elif task==4:
                import SplitAnnotations
                avianz = SplitAnnotations.SplitData()

            # catch bad initialiation
            if avianz:
                avianz.activateWindow()
            else:
                return

            out = app.exec_()
            QApplication.closeAllWindows()
            QApplication.processEvents()

            # catch exit code to see if restart requested:
            # (note: do not use this for more complicated cleanup,
            #  no guarantees that it is returned before program closes)
            if out == 0:
                # default quit
                break
            elif out == 1:
                # restart to splash screen
                task = None
            elif out == 2:
                # request switch to Splitter
                task = 4


try:
    mainlauncher()
except Exception:
    import traceback
    print(traceback.format_exc())
    input("Encountered error. Report it with the text above to AviaNZ team at www.avianz.net.\nPress ENTER to exit")
    raise
