@echo off
REM Navigate to the folder containing the Streamlit app
cd "%~dp0"

REM Ensure Python and Streamlit are accessible
python -m streamlit run gui.py