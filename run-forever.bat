@echo off
:: CryptoWorm 24/7 Runner - restarts on crash
:: Launched via Task Scheduler at system startup

cd /d C:\Users\simon\code\cryptoworm

:loop
echo [%date% %time%] Starting CryptoWorm...
py -3 bot.py
echo [%date% %time%] CryptoWorm exited with code %errorlevel%. Restarting in 30s...
timeout /t 30 /nobreak >nul
goto loop
