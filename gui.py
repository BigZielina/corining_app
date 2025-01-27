
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
import itertools

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


    #connectors should actually be an even number
    n_connectors = st.number_input("Number of Connectors", min_value=2, max_value=100, value=10, step=2)
    n_wavelengths = st.number_input("Number of Wavelengths", min_value=1, max_value=10, value=1, step=1)
    n_fibers = st.number_input("Number of Fibers", min_value=1, max_value=50, value=1, step=1)

    st.write("Wavelength Selection")
    st.markdown("Select predefined wavelengths or add your own. For custom wavelengths, separate values with semicolons (e.g., 1309.45;1789;1).")

    predefined_wavelengths = ["1310", "1490", "1550", "1625"]
    selected_wavelengths = [
        wavelength for wavelength in predefined_wavelengths
        if st.checkbox(wavelength, value=True)
    ]

    user_defined_wavelengths = st.text_input(
        "Enter additional wavelengths (separate with semicolons):",
        value=""
    )
    if user_defined_wavelengths.strip():
        additional_wavelengths = user_defined_wavelengths.split(";")
        additional_wavelengths = [w.strip() for w in additional_wavelengths if w.strip()]
    else:
        additional_wavelengths = []

    wavelengths = selected_wavelengths + additional_wavelengths #list of wavelenghs for template generating


    file_name = st.text_input("File Name", value="template.xlsx")

    buffer = BytesIO()

    #TODO : język jako parametr w GUI żeby kontrolować język instrukcji, błędów i najlepiej całego UI
    app_language = "english"
    data_core_instance = DataCore(language = app_language)

    # TODO : trzeba zmienić żeby gui dawało liste długości fal które mają być w pliku
    # i tak żeby wszystkie te bazowe ([1310, 1550,...]) były do zaznaczenia jakoś tak
    # [] 1310nm
    # [] 1550mn
    # no i potem pole na potencjalne dodatkowe wartośći
    #
    # warnings = wavelengths = [1310, 1550]
    data_core_instance.create_excel_template(
        n_connectors=n_connectors,
        path=buffer,
        wavelengths=wavelengths,
        n_fibers=n_fibers
    )

    buffer.seek(0)
    if wavelengths:
        st.success(f"The following wavelengths are included in the template: {', '.join(wavelengths)}")
    else:
        st.warning("No wavelengths selected.")

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

uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

if uploaded_file is not None:

    #TODO teraz load_excel zwraca liste ostrzeżeń jako stringów które trzeba jakoś ludzią pokazać
    warnings = data_core_instance.load_excel(uploaded_file)


    if len(warnings) != 0:
        st.warning("\n".join(warnings))

    dfs1,n_jumpers = generate_df(uploaded_file)

    with st.expander("RM number of jumper choices"):
        st.write("Provide the desired number of jumpers to be chosen in each random mating")
        selected_connector_number = st.number_input(
            "Number of choices",
            min_value=1,
            max_value=n_jumpers,
            value=n_jumpers-1 if n_jumpers > 1 else 1,
            step=1
        )
    st.subheader("Overview Table")
    st.write(dfs1[0])
    max_selected_connectors = n_jumpers
    plots, dfs = generate_plots(uploaded_file,selected_connector_number)
    tab_titles,tab_categories = generate_tab_titles()

    dfs = list(dfs)
    dfs.insert(0, dfs1[0])
    dfs = tuple(dfs)

    all_plots = save_plots_to_bytes(plots)
    all_data_zip = create_zip_file(dfs, all_plots)
    st.download_button(
        label="Download All Data (Tables and Plots)",
        data=all_data_zip,
        file_name="all_data.zip",
        mime="application/zip"
    )

    tab_cats = st.tabs(tab_categories)
    i_global = 0
    for j,tab_cat in enumerate(tab_cats):

        with tab_cat:
            tabs = st.tabs(tab_titles[j])
            for i, tab in enumerate(tabs):
                with tab:

                    st.subheader(f"{tab_titles[j][i]}")

                    st.pyplot(plots[i_global])

                    buf = BytesIO()
                    plots[i_global].savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button(
                        label=f"Download Plot {i+1}",
                        data=buf,
                        file_name=f"plot_{i+1}.png",
                        mime="image/png",
                        key=f"plot{j}{i_global}{i}"
                    )
                    buf.close()

                    if i_global+1 < len(dfs)+1:  # Check if DataFrame exists
                        st.write(dfs[i_global+1])

                        csv_data = convert_df_to_csv(dfs[i_global+1])
                        st.download_button(
                            label=f"Download Table {i+1} as CSV",
                            data=csv_data,
                            file_name=f"table_{i+1}.csv",
                            mime="text/csv",
                            key=f"{j}{i_global}{i}"
                    )
                    i_global += 1
    print(tab_titles)
    print(list(itertools.chain(*tab_titles)))
    st.download_button(
        label="Generate and Download PDF Report",
        data=save_pdf(create_pdf(list(itertools.chain(*tab_titles)), plots, dfs)),
        file_name="RM_report.pdf",
        mime="application/pdf",
    )
