@echo off
setlocal

cd /d "%~dp0"

echo Starting llama-webui...
echo Model discovery: project models, %%USERPROFILE%%\models, %%USERPROFILE%%\llama-rpc\models, and Hugging Face cache
echo.

.venv\Scripts\llama-webui.exe
