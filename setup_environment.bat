@echo off
REM Setup script for AusLAMP_GA project
REM Run this once to create the conda environment and register Jupyter kernel

echo ============================================================
echo AusLAMP Victoria - Environment Setup
echo ============================================================
echo.

echo [1/3] Creating conda environment from environment.yml...
call conda env create -f environment.yml
if errorlevel 1 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)
echo.

echo [2/3] Activating environment...
call conda activate AusLAMP_GA
if errorlevel 1 (
    echo ERROR: Failed to activate environment
    pause
    exit /b 1
)
echo.

echo [3/3] Registering Jupyter kernel...
python -m ipykernel install --user --name=AusLAMP_GA --display-name="Python (AusLAMP_GA)"
if errorlevel 1 (
    echo ERROR: Failed to register kernel
    pause
    exit /b 1
)
echo.

echo ============================================================
echo Setup complete!
echo ============================================================
echo.
echo Next steps:
echo   1. cd notebooks
echo   2. jupyter notebook
echo   3. Open 01_metadata_exploration.ipynb
echo   4. Select: Kernel -^> Change kernel -^> Python (AusLAMP_GA)
echo.
pause
