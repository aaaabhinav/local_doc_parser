@echo off
echo ==============================================
echo Installing API Dependencies...
echo ==============================================
echo.
pip install -r requirements.txt
echo.
echo ==============================================
echo Installing pre-compiled Llama-CPP (No C++ compiler needed)...
echo ==============================================
echo.
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
echo.
echo ==============================================
echo Installation Complete!
echo ==============================================
pause
