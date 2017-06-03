# This is the main window for AviaNZ program with three options

# 31/05/17
# Author: Nirosha Priyadarshani

from PyQt4.QtGui import *
import sys

import interface_pyqtgraph
import interface_FindSpecies
import Dialogs

# Start the application
app = QApplication(sys.argv)

# This screen asks what you want to do, then gets the response
first = Dialogs.StartScreen()
first.setWindowIcon(QIcon('img/AviaNZ.ico'))
first.show()
app.exec_()

task = first.getValues()
# print task

if task == 1:
    avianz = interface_pyqtgraph.AviaNZInterface(configfile='AviaNZconfig.txt')
    avianz.setWindowIcon(QIcon('img/AviaNZ.ico'))
    avianz.show()
    app.exec_()
elif task==2:
    avianz = interface_FindSpecies.AviaNZFindSpeciesInterface(configfile='AviaNZconfig.txt')
    avianz.setWindowIcon(QIcon('img/AviaNZ.ico'))
    avianz.show()
    app.exec_()
else:
    app.exit()
