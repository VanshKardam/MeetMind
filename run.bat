@echo off
echo =======================================================
echo 🧠 Starting MeetMind AI Assistant...
echo =======================================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Launching Streamlit dashboard...
streamlit run app.py

pause