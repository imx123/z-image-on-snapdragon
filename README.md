# Z-Image on Snapdragon 8 Elite

This repository is the Android-side bring-up scaffold for running Z-Image-Turbo on Snapdragon 8 Elite.

## Target execution split

- Transformer: Adreno GPU (FP16 first, then mixed INT4/FP16)
- Text encoder: Qualcomm QNN HTP/NPU when the exported graph is supported
- VAE: Adreno GPU FP16 initially
- Scheduler: C++ CPU implementation

The first milestone is a single 512x512, batch-one, 8-step Z-Image-Turbo path. Model weights and Qualcomm SDK binaries are intentionally not checked in.

## Layout

- android/: Android Studio project and JNI runtime shell
- tools/: export and graph validation scripts
- docs/: bring-up notes and backend acceptance criteria

## Local prerequisites

- Android Studio with NDK and CMake
- Python 3.10+
- A local checkout of the official Z-Image repository
- Qualcomm AI Engine Direct / QNN SDK for Snapdragon 8 Elite bring-up

Model access is ModelScope-first. The default model is Tongyi-MAI/Z-Image-Turbo and the download helper uses ModelScope `local_dir`; Hugging Face is not part of the default path. The current local snapshot is registered in `config/modelscope-z-image-turbo.json`.

See docs/bring-up.md for the staged workflow.

QAIRT 2.49.0.260730 is installed at `C:\Qualcomm\qairt\qairt\2.49.0.260730`. Run `tools/setup_qairt_env.ps1` in each new PowerShell session before conversion commands.

If PowerShell execution policy blocks scripts, use `Set-ExecutionPolicy -Scope Process Bypass` for the current terminal only, then dot-source the setup script: `. .\tools\setup_qairt_env.ps1`.

QAIRT uses the isolated runtime `C:\Qualcomm\qairt\runtime\python312\python.exe`; it is separate from the system Python installation.
