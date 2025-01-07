from reportlab.platypus import Paragraph,Image,Table, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

import matplotlib.pyplot as plt
from io import BytesIO
from svglib.svglib import svg2rlg
import pandas as pd


def prepare_table(table : pd.DataFrame) -> Table:
    cols = table.columns
    tableT = table.T.to_numpy()
    for i, row in enumerate(tableT):
        row = cols[i] + row

    t=Table(tableT,style=[
                ('GRID',(0,0),(-1,-1),0.5,colors.black),
                ])
    return t

def prepare_plot(plot : plt.subplots) -> Image:
    imgdata = BytesIO()
    plot.savefig(imgdata, format='svg')
    imgdata.seek(0)  # rewind the data

    drawing=svg2rlg(imgdata)
    img = Image(drawing)

    return img


def create_pdf(names, plots, tables):
    
    styleSheet=getSampleStyleSheet()

    story = [Paragraph('''
                <b>A RM test raport</b>''',
                styleSheet["Heading1"])]
    
    for i, name, plot, table in enumerate(names, plots, tables):

        story.append(Paragraph(name,styleSheet["Heading3"]))
        story.append(prepare_plot(plot))
        story.append(prepare_table(table))
        
    doc = SimpleDocTemplate("tab.pdf", pagesize = A4)

    return