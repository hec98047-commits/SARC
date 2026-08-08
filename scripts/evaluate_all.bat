@echo off
setlocal
if "%~6"=="" (echo Usage: %~nx0 ^<mvtec_root^> ^<visa_root^> ^<fgclip_model^> ^<sarc_model^> ^<mvtec_output_root^> ^<visa_output_root^>& exit /b 2)
for %%S in (1 2 4) do (
  call "%~dp0_run_fewshot.bat" mvtec %%S "%~1" "%~3" "%~4" "%~5\%%Sshot" || exit /b 1
  call "%~dp0_run_fewshot.bat" visa %%S "%~2" "%~3" "%~4" "%~6\%%Sshot" || exit /b 1
)
