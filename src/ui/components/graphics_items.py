# coding=latin-1

# graphics_items.py
# Graphics items for the AviaNZ program


# Version 3.4 18/12/24
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

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

from PyQt6.QtWidgets import QGraphicsRectItem


class ClickableRectItem(QGraphicsRectItem):
    # QGraphicsItem doesn't include signals, hence this mess
    def __init__(self, *args, **kwds):
        QGraphicsRectItem.__init__(self, *args, **kwds)

    def mousePressEvent(self, ev):
        super(ClickableRectItem, self).mousePressEvent(ev)
        # send the position of this rectangle in ViewBox coords
        # left corner:
        # x = self.mapRectToParent(self.boundingRect()).x()
        # or center:
        x = self.mapRectToParent(self.boundingRect()).center().x()
        self.parentWidget().resend(x)