@echo off
cd "%~dp0"

cxFreeze --script run.py --target-dir build\rm_app

copy *py .\build\rm_app