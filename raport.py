from reportlab.platypus import Paragraph,Image,Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm

import matplotlib.pyplot as plt
from io import BytesIO
from svglib.svglib import svg2rlg
import pandas as pd
import numpy as np

import itertools


def prepare_table(table : pd.DataFrame) -> Table:
    if table is 0:
        return
    tablist = []
    cols = table.columns
    #print(cols)
    #print(table)
    tableT = table.T.to_numpy()
    
    if tableT.shape[1] > 30:
        T = np.array_split(tableT, 5, axis=1)
    elif tableT.shape[1] > 5:
        T = np.array_split(tableT, 3, axis=1)        
    else:
        T = [tableT]
    for t in T:
        
        if t.shape[1] == 1:
            t = list(map(list, t))
            t = map(np.round, t)
            t = [cols, t] 
        else:
            t = list(map(list, t))
            t = list(map(np.round, t))
            
            for i, row in enumerate(t):
                row = row.tolist()
                row.insert(0, cols[i])

        t=Table(t,style=[
                    ('GRID',(0,0),(-1,-1),0.5,colors.black),
                    ("LINEABOVE", (0,0),(-1,0),1,colors.black)
                    ])
        tablist.append(t)
    return tablist

def prepare_plot(plot : plt.subplots) -> Image:
    if plot == 0:
        return
    imgdata = BytesIO()
    plot.savefig(imgdata, format='svg')
    imgdata.seek(0)  # rewind the data

    drawing=svg2rlg(imgdata)
    img = Image(drawing, width=19*cm, height=8.5*cm)

    return img


def create_pdf(names, plots, tables):
    
    styleSheet=getSampleStyleSheet()

    Ltables = list(tables)
    
    t=Table(list(map(list, Ltables.pop(0).to_numpy())),style=[
            ('GRID',(0,0),(-1,-1),0.5,colors.black),
            ])

    story = [Paragraph('''
                <b>A RM test raport</b>''',
                styleSheet["Heading1"]), t]

    for name, plot, table in itertools.zip_longest(names, plots, Ltables, fillvalue=0):

        print(name)
        story.append(Paragraph(name,styleSheet["Heading3"]))
        story.append(prepare_plot(plot))
        for i in prepare_table(table):
            story.append(i)
        story.append(PageBreak())

    return story