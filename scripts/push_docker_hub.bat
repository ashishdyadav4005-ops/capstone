@echo off
setlocal

if "%~1"=="" (
    set /p USERNAME="Enter your Docker Hub username: "
) else (
    set USERNAME=%~1
)

if "%USERNAME%"=="" (
    echo Error: Docker Hub username is required.
    exit /b 1
)

python scripts\push_docker_hub.py %USERNAME% %2 %3 %4 %5
