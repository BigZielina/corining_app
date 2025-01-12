import streamlit
import streamlit.runtime.scriptrunner.magic_funcs
from reportlab import platypus
import numpy
import pandas 
import xlsxwriter
import reportlab
import regex
import scipy
import matplotlib
import openpyxl
import svglib
from svglib import svglib
import streamlit.web.cli as stcli
import os, sys


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("gui.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())