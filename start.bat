call venv\Scripts\activate.bat
set TORCH_HOME=.\.torch_cache
set /p MODEL_TYPE=Enter model type (small, large, hybrid): 
python run_midas.py input.jpg --model %MODEL_TYPE%
pause