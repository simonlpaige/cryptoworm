@echo off
:: CryptoBot 24/7 Runner — restarts on crash
:: Launched via Task Scheduler at system startup

cd /d C:\Users\simon\.openclaw\workspace\crypto-bot

:loop
echo [%date% %time%] Starting CryptoBot...
py -3 bot.py
echo [%date% %time%] CryptoBot exited with code %errorlevel%. Restarting in 30s...
timeout /t 30 /nobreak >nul
goto loop
