#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUTHOR Romain Giot <giot.romain@gmail.com>

The aim of this library is to read the Purdue FTA database.
"""

from odf.opendocument import load

class PurdueFTAReader(object):
    """Reader class of the Purdue FTA database.
    """

    def __init__(self, fname='Purdue-FTA.ods'):
        """Open the ODS file.

        @param fname : Path of the file.
        """

        load(fname)

if __name__ == "__main__":
    reader = PurdueFTAReader()

