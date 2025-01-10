from reportlab.platypus import Paragraph,Image,Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm

import matplotlib.pyplot as plt
from io import BytesIO
from svglib.svglib import svg2rlg
import pandas as pd
import numpy as np



def prepare_table(table : pd.DataFrame) -> Table:
    tablist = []
    cols = table.columns

    tableT = table.T.to_numpy()
    try:
        tableT = tableT.round(3)
    except:
        pass
    if tableT.shape[1] > 5:
        T = np.array_split(tableT, 3, axis=1)        
    else:
        T = [tableT]
    for t in T:
        t = list(map(list, t))
        
        for i, row in enumerate(t):

            row.insert(0, cols[i])

        t=Table(t,style=[
                    ('GRID',(0,0),(-1,-1),0.5,colors.black),
                    ("LINEABOVE", (0,0),(-1,0),1,colors.black)
                    ])
        tablist.append(t)
    return tablist

def prepare_plot(plot : plt.subplots) -> Image:
    imgdata = BytesIO()
    plot.savefig(imgdata, format='svg')
    imgdata.seek(0)  # rewind the data

    drawing=svg2rlg(imgdata)
    img = Image(drawing, width=15*cm, height=10*cm)

    return img


def create_pdf(names, plots, tables):
    
    styleSheet=getSampleStyleSheet()

    Ltables = list(tables)
    
    t=prepare_table(Ltables.pop(0))[0]

    story = [Paragraph('''
                <b>A RM test report</b>''',
                styleSheet["Heading1"]), t]

    for name, plot, table in zip(names, plots, Ltables):


        story.append(Paragraph(name,styleSheet["Heading3"]))
        story.append(prepare_plot(plot))
        
        for i in prepare_table(table):
            story.append(i)
        story.append(PageBreak())

    return story