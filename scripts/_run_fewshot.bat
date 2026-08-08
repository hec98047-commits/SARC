@echo off
setlocal
if "%~6"=="" (echo Usage: %~nx0 ^<mvtec^|visa^> ^<1^|2^|4^> ^<data_root^> ^<fgclip_model^> ^<sarc_model^> ^<output_dir^>& exit /b 2)
python src\sarc\run_sarc_protocol.py --dataset "%~1" --data_root "%~3" --model_path "%~4" --sarc_model_path "%~5" --output_dir "%~6" --shots_list "%~2" --seeds 42
exit /b %errorlevel%
