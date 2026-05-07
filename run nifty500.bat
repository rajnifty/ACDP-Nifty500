@echo off
title ACDP Nifty 500 Analytics Engine

echo ===================================================
echo   Starting ACDP Nifty 500 Dashboard...
echo   Please wait while the engine initializes.
echo ===================================================
echo.

:: Using "python -m" bypasses the Windows PATH issue
python -m streamlit run nifty500.py

:: Keep the window open if there's an error
pause