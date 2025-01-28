
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


app_language = st.sidebar.selectbox("Language/Język",["english","polish"])


expander_str = "Generate Excel Template"
if app_language == "polish":
    expander_str = "Wygeneruj szablon excela"
with st.expander(expander_str):
    
    if app_language == "english":
        st.write("Set the parameters below to create your Excel template.")
    if app_language == "polish":
        st.write("Ustaw poniższe parametry, aby utworzyć szablon w programie Excel.")



    #connectors should actually be an even number

    if app_language == "english":
        n_connectors = st.number_input("Number of Connectors", min_value=2, max_value=100, value=10, step=2)
        # n_wavelengths = st.number_input("Number of Wavelengths", min_value=1, max_value=10, value=1, step=1)
        n_fibers = st.number_input("Number of Fibers", min_value=1, max_value=50, value=1, step=1)
 
    if app_language == "polish":
        n_connectors = st.number_input("Liczba złączek", min_value=2, max_value=100, value=10, step=2)
        # n_wavelengths = st.number_input("Liczba długości fali", min_value=1, max_value=10, value=1, step=1)
        n_fibers = st.number_input("Liczba włókien", min_value=1, max_value=50, value=1, step=1)
    
    if app_language == "english":
        st.write("Wavelength Selection")
        st.markdown("Select predefined wavelengths or add your own. For custom wavelengths, separate values with semicolons (e.g., 1309.45;1789;1).")
    if app_language == "polish":
        st.write("Wybór długości fali")
        st.markdown("Wybierz wstępnie zdefiniowane długości fal lub dodaj własne. W przypadku niestandardowych długości fal rozdziel wartości średnikami (np. 1309.45;1789;1).")

    predefined_wavelengths = ["1310", "1490", "1550", "1625"]
    selected_wavelengths = [
        wavelength for wavelength in predefined_wavelengths
        if st.checkbox(wavelength, value=True)
    ]


    text_input_string = "Enter additional wavelengths (separate with semicolons):"
    if app_language == "polish":
        text_input_string = "Wprowadź dodatkowe długości fal (oddzielone średnikami):"

    user_defined_wavelengths = st.text_input(
        text_input_string,
        value=""
    )
    if user_defined_wavelengths.strip():
        additional_wavelengths = user_defined_wavelengths.split(";")
        additional_wavelengths = [w.strip() for w in additional_wavelengths if w.strip()]
    else:
        additional_wavelengths = []

    wavelengths = selected_wavelengths + additional_wavelengths #list of wavelenghs for template generating


    if app_language == "english":
        file_name = st.text_input("File Name", value="template.xlsx")
    if app_language == "polish":
        file_name = st.text_input("Nazwa pliku", value="szablon.xlsx")

    buffer = BytesIO()

    data_core_instance = DataCore(language = app_language)


    data_core_instance.create_excel_template(
        n_connectors=n_connectors,
        path=buffer,
        wavelengths=wavelengths,
        n_fibers=n_fibers
    )

    buffer.seek(0)

    if app_language == "english":
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

    if app_language == "polish":
        if wavelengths:
            st.success(f"W szablonie uwzględniono następujące długości fal: {', '.join(wavelengths)}")
        else:
            st.warning("Nie wybrano żadnych długości fal.")
    
        st.download_button(
            label="Wygeneruj i pobierz szablon Excela",
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
if app_language == "english":
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])
if app_language == "polish":
    uploaded_file = st.file_uploader("Prześlij swój plik Excel", type=['xlsx'])


if uploaded_file is not None:

    #TODO teraz load_excel zwraca liste ostrzeżeń jako stringów które trzeba jakoś ludzią pokazać
    warnings = data_core_instance.load_excel(uploaded_file)


    if len(warnings) != 0:
        st.warning("\n".join(warnings))

    dfs1,n_jumpers = generate_df(uploaded_file)
    if app_language == "english":
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
        
    if app_language == "polish":
        with st.expander("liczba wyborów kabli do RM"):
            st.write("Podaj żądaną liczbę kabli, które mają zostać wybrane w każdym losowym połączeniu")
            selected_connector_number = st.number_input(
                "liczba wyborów",
                min_value=1,
                max_value=n_jumpers,
                value=n_jumpers-1 if n_jumpers > 1 else 1,
                step=1
            )
        st.subheader("Ogólne dane")

    st.write(dfs1[0])
    max_selected_connectors = n_jumpers
    plots, dfs = generate_plots(uploaded_file,selected_connector_number,language=app_language)
    tab_titles,tab_categories = generate_tab_titles(app_language)

    dfs = list(dfs)
    dfs.insert(0, dfs1[0])
    dfs = tuple(dfs)

    all_plots = save_plots_to_bytes(plots)
    all_data_zip = create_zip_file(dfs, all_plots)
    if app_language == 'english':
        st.download_button(
            label="Download All Data (Tables and Plots)",
            data=all_data_zip,
            file_name="all_data.zip",
            mime="application/zip"
        )

    if app_language == 'polish':
        st.download_button(
            label="Pobierz wszystkie dane (tabele i wykresy)",
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
                    if app_language == "english":
                        st.download_button(
                            label=f"Download Plot {i+1}",
                            data=buf,
                            file_name=f"plot_{i+1}.png",
                            mime="image/png",
                            key=f"plot{j}{i_global}{i}"
                        )

                    if app_language == "polish":
                        st.download_button(
                            label=f"Pobierz wykres {i+1}",
                            data=buf,
                            file_name=f"plot_{i+1}.png",
                            mime="image/png",
                            key=f"plot{j}{i_global}{i}"
                        )
                    buf.close()

                    if i_global+1 < len(dfs)+1:  # Check if DataFrame exists
                        st.write(dfs[i_global+1])

                        csv_data = convert_df_to_csv(dfs[i_global+1])

                        if app_language == "english":
                            st.download_button(
                                label=f"Download Table {i+1} as CSV",
                                data=csv_data,
                                file_name=f"table_{i+1}.csv",
                                mime="text/csv",
                                key=f"{j}{i_global}{i}"
                            )
                        if app_language == "polish":
                            st.download_button(
                                label=f"Pobierz tabele {i+1} jako CSV",
                                data=csv_data,
                                file_name=f"table_{i+1}.csv",
                                mime="text/csv",
                                key=f"{j}{i_global}{i}"
                            )
                    i_global += 1

    if app_language == "english":

        st.download_button(
            label="Generate and Download PDF Report",
            data=save_pdf(create_pdf(list(itertools.chain(*tab_titles)), plots, dfs)),
            file_name="RM_report.pdf",
            mime="application/pdf",
        )

    if app_language == "polish":

        st.download_button(
            label="Wygeneruj i pobierz raport PDF",
            data=save_pdf(create_pdf(list(itertools.chain(*tab_titles)), plots, dfs)),
            file_name="RM_report.pdf",
            mime="application/pdf",
        )
