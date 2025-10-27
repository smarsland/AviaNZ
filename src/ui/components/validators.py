
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

# Validation for the filters

from PyQt6.QtGui import QValidator
from PyQt6.QtCore import Qt

class FiltValidator(QValidator):
    def __init__(self, list_files, check_reserved_m=False):
        super().__init__()
        self.listFiles = list_files
        self.check_reserved_m = check_reserved_m

    def validate(self, input, pos):
        if not input.endswith('.txt'):
            input = input + '.txt'
        if input == ".txt" or input == "":
            return (QValidator.State.Intermediate, input, pos)
        if self.check_reserved_m and input == "M.txt":
            print("filter name \"M\" reserved for manual annotations")
            return (QValidator.State.Intermediate, input, pos)
        if self.listFiles.findItems(input, Qt.MatchFlag.MatchExactly):
            print("duplicated input", input)
            return (QValidator.State.Intermediate, input, pos)
        else:
            return (QValidator.State.Acceptable, input, pos)
