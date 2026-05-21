@echo off
echo ============================================
echo  OULAD Early Warning System - Starting...
echo ============================================
echo.

cd /d "%~dp0"

echo [1/2] Starting FastAPI (port 8000)...
echo       Loading models + data may take ~30s on first start.
start "EWS-API" cmd /k "cd /d %~dp0 && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

echo Waiting for API to initialize (30s)...
timeout /t 30 /nobreak >nul

echo [2/2] Starting Streamlit (port 8501)...
start "EWS-UI" cmd /k "cd /d %~dp0 && python -m streamlit run ui/app.py --server.port 8501"

echo.
echo ============================================
echo  API docs: http://localhost:8000/docs
echo  Demo UI:  http://localhost:8501
echo ============================================
