@echo off
echo ============================================
echo  LocalMind AI — Starting Model Servers
echo ============================================
echo.

cd /d "%~dp0backend"
call venv\Scripts\activate

echo [1/1] Starting DeepSeek R1 1.5B on port 8003...
start "DeepSeek-R1 :8003" python -m llama_cpp.server ^
  --model models/deepseek.gguf ^
  --port 8003 ^
  --host 0.0.0.0 ^
  --n_ctx 2048 ^
  --n_threads 4

echo.
echo Models started! Waiting for them to load (~15s)...
echo.
echo   DeepSeek R1 1.5B  ->  http://localhost:8003
echo.
echo You can now start the backend:
echo   cd backend
echo   venv\Scripts\activate
echo   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
pause
