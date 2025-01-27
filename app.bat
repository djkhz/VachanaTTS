@echo off

REM Save the current directory
set "current_dir=%CD%"

REM Change to the location of Python executable
cd /d F:\RVC\VachanaTTS\venv\Scripts

REM Activate the Python environment
call activate

REM Change to the location of the Python script
cd /d F:\RVC\VachanaTTS

REM Run the Python script
python app.py

REM Return to the original directory
cd /d %current_dir%

REM Pause to keep the command prompt window open (optional)
pause