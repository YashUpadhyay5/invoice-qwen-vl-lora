@echo off
echo ========================================================
echo     Publishing Invoice Qwen VL LoRA to Hugging Face
echo ========================================================
python "%~dp0upload_to_hf.py" %*
pause
