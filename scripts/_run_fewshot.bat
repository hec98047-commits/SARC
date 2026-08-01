@echo off
setlocal
if "%~6"=="" (echo Usage: %~nx0 ^<mvtec^|visa^> ^<1^|2^|4^> ^<data_root^> ^<fgclip_ckpt^> ^<mg_ckpt^> ^<output_dir^>& exit /b 2)
python src\pgcre_fgclip\run_fewshot_protocol.py --dataset "%~1" --data_root "%~3" --model_path "%~4" --mg_model_path "%~5" --output_dir "%~6" --shots_list "%~2" --seeds 42 --methods ours
exit /b %errorlevel%
