
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
from data_loading import DataCore, generate_df  # Import DataCore from data_loading.py
from plots import generate_plots,generate_tab_titles
from raport import create_pdf
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import letter
import os


st.set_page_config(layout="wide")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def convert_dfs_to_excel(dfs):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for i, df in enumerate(dfs):
            df.to_excel(writer, sheet_name=f"Table {i+1}", index=False)
    output.seek(0)
    return output

def save_plots_to_bytes(plots):
    plot_bytes = []
    for i, plot in enumerate(plots):
        buf = BytesIO()
        plot.savefig(buf, format="png")
        buf.seek(0)
        plot_bytes.append((f"plot_{i+1}.png", buf.read()))
        buf.close()
    return plot_bytes

def create_zip_file(tables, plots):
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as zf:
        # Add tables
        for i, table in enumerate(tables):
            csv_data = convert_df_to_csv(table)
            zf.writestr(f"table_{i+1}.csv", csv_data)
        # Add plots
        for filename, data in plots:
            zf.writestr(filename, data)
    output.seek(0)
    return output

def save_pdf(story):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize = letter)
    doc.build(story)
    buffer.seek(0)
    return buffer




#--------------------------------------------------------------------------

st.title("Data Viewer")

with st.expander("Generate Excel Template"):
    st.write("Set the parameters below to create your Excel template.")
    
    n_connectors = st.number_input("Number of Connectors", min_value=1, max_value=100, value=5, step=1)
    n_wavelengths = st.number_input("Number of Wavelengths", min_value=1, max_value=10, value=1, step=1)
    n_fibers = st.number_input("Number of Fibers", min_value=1, max_value=50, value=1, step=1)
    file_name = st.text_input("File Name", value="template.xlsx")
    
    buffer = BytesIO()

    data_core_instance = DataCore()
    data_core_instance.create_excel_template(
        n_connectors=n_connectors,
        path=buffer,  
        n_wavelengths=n_wavelengths,
        n_fibers=n_fibers
    )

    buffer.seek(0)  
    st.download_button(
        label="Generate and Download Excel Template",
        data=buffer,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# template_csv = "template.xlsx"
# st.download_button(
#     label="Download Template",
#     data=open(template_csv, 'rb').read(),
#     file_name="template.xlsx"
# )
max_selected_connectors = 100
with st.expander("Select Connector Number"):
    st.write("Provide the desired connector number below.")
    selected_connector_number = st.number_input(
        "Connector Number",
        min_value=1,
        max_value=max_selected_connectors,
        value=10,
        step=1
    )
uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

if uploaded_file is not None:
    dfs,n_jumpers = generate_df(uploaded_file,selected_connector_number)
    max_selected_connectors = n_jumpers
    plots = generate_plots(uploaded_file)
    tab_titles = generate_tab_titles()
    st.subheader("Overview Table")
    st.write(dfs[0])

    all_plots = save_plots_to_bytes(plots)
    all_data_zip = create_zip_file(dfs, all_plots)
    st.download_button(
        label="Download All Data (Tables and Plots)",
        data=all_data_zip,
        file_name="all_data.zip",
        mime="application/zip"
    )

    # tab_titles = [
    #     "Combinations of connectors for all wavelengths",
    #     "All wavelengths",
    #     "Reference connectors",
    #     "dut connectors",
    #     "Mean Jumpers",
    #     "Jumpers Std",
    #     "Mean 1550",
    #     "Mean 1650",
    #     "Mean 1770",
    #     "97th 1550",
    #     "97th 1650",
    #     "97th 1770",
    #     "Mean Connectors",
    #     "Connectors Std"
    # ]


    tabs = st.tabs(tab_titles)

    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"{tab_titles[i]}")
            
            st.pyplot(plots[i])
            
            buf = BytesIO()
            plots[i].savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label=f"Download Plot {i+1}",
                data=buf,
                file_name=f"plot_{i+1}.png",
                mime="image/png"
            )
            buf.close()

            if i + 1 < len(dfs):  # Check if DataFrame exists
                st.write(dfs[i+1])

                csv_data = convert_df_to_csv(dfs[i+1])
                st.download_button(
                    label=f"Download Table {i+1} as CSV",
                    data=csv_data,
                    file_name=f"table_{i+1}.csv",
                    mime="text/csv"
            )


    st.download_button(
        label="Generate and Download PDF Raport",
        data=save_pdf(create_pdf(tab_titles, plots, dfs)),
        file_name="RM_report.pdf",
        mime="application/pdf",
    )