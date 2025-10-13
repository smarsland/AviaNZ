
# Version 4.1 09/10/25
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

# Message popup abstraction for core/UI separation

QtMM = False

class MessagePopup:
    """Default console-based MessagePopup used when no GUI is registered.

    Existing core code expects to be able to call MessagePopup(level, title, text)
    and optionally call exec() on the result. To avoid importing UI modules
    at core import time, this default implementation simply prints to stdout.
    """
    def __init__(self, level, title, text):
        self.level = level
        self.title = title
        self.text = text

    def exec(self):
        print(f"{self.level}:{self.title}:{self.text}")


def set_message_popup_class(cls):
    """Register a MessagePopup implementation provided by the UI layer.

    This rebinds the global name MessagePopup to the provided class so existing
    code that calls MessagePopup(...) continues to work. The UI layer should
    call this during application startup after importing GUI modules.
    """
    global MessagePopup, QtMM
    MessagePopup = cls
    QtMM = True


def get_message_popup(*args, **kwargs):
    """Factory helper to instantiate the currently-registered MessagePopup."""
    return MessagePopup(*args, **kwargs)
