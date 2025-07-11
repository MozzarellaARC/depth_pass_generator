call venv\Scripts\activate.bat
set /p MODEL_TYPE=Enter model type (small, large, hybrid): 
python run_midas.py input.jpg --model %MODEL_TYPE%
pause