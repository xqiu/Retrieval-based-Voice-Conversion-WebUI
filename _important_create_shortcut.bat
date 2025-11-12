@echo off
setlocal

set "EXE=.venv\python.exe"
if not exist %EXE% (
  echo set python venv path
  set "EXE=venv\Scripts\python.exe"
)

set "TARGET=run_infer_web.py"

:: Get the absolute path of the current directory
set "current_dir=%~dp0"
set "exe_path=%current_dir%%EXE%"

:: Get the Desktop path for the current user
set "desktop=%USERPROFILE%\Desktop"

:: Define shortcut name
set "shortcut_name=Run RVC Voice Cloning.lnk"
set "shortcut_path=%desktop%\%shortcut_name%"

if exist "%shortcut_path%" (
  del "%shortcut_path%"
)

:: Create a temporary VBScript to generate the shortcut
set "vbs_file=%temp%\create_shortcut.vbs"
if exist "%vbs_file%" (
  del "%vbs_file%"
)

echo Set oWS = WScript.CreateObject("WScript.Shell") >> "%vbs_file%"
echo sLinkFile = "%shortcut_path%" >> "%vbs_file%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%vbs_file%"
echo oLink.TargetPath = "%exe_path%" >> "%vbs_file%"
echo oLink.Arguments = "%TARGET%" >> "%vbs_file%"
echo oLink.WorkingDirectory = "%current_dir%" >> "%vbs_file%"
echo oLink.WindowStyle = 1 >> "%vbs_file%"
echo oLink.IconLocation = "%exe_path%,0" >> "%vbs_file%"
echo oLink.Description = "Shortcut to RVC" >> "%vbs_file%"
echo oLink.Save >> "%vbs_file%"

:: Run the VBScript
cscript //nologo "%vbs_file%"

:: Cleanup
del "%vbs_file%"

echo Shortcut created on desktop: %shortcut_path%
pause