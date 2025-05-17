# Troubleshooting: ffmpeg PATH Issue in facebeak

## Problem

When running facebeak for video processing, you may encounter the following warning:

```
[WARNING] ffmpeg is not installed or not found in PATH.
Audio transfer and video compression will not work.
```

This results in output videos missing audio, and disables video compression features.

## Cause

- ffmpeg is not installed, or
- ffmpeg is installed, but its `bin` directory is not included in the Windows system PATH, or
- The PATH was updated, but running applications (Python, GUI, terminals) have not picked up the change because they were not restarted.

## Solution

### 1. Install ffmpeg
- Download the latest static build from https://ffmpeg.org/download.html
- Extract it to a folder, e.g., `C:\ffmpeg\ffmpeg-7.1.1-full_build\bin`

### 2. Add ffmpeg to the System PATH
- Open the Start Menu and search for "Environment Variables"
- Click "Edit the system environment variables"
- In the System Properties window, click "Environment Variables..."
- Under "System variables", find and select `Path`, then click Edit
- Click New and add the path to your ffmpeg `bin` directory, e.g.:
  ```
  C:\ffmpeg\ffmpeg-7.1.1-full_build\bin
  ```
- Click OK on all dialogs to save

### 3. Restart Your Computer
- **Important:** Any open applications (including Python, VSCode, terminals, or the facebeak GUI) will not see the updated PATH until you restart them.
- For best results, restart your computer or log out and log back in.

### 4. Verify ffmpeg is Available
- Open a new Command Prompt or PowerShell window
- Run:
  ```
  ffmpeg -version
  ```
- You should see version information. If you get an error, repeat the steps above.

### 5. Re-run facebeak
- Launch the facebeak GUI or run your processing script again
- The warning should be gone, and output videos should include audio

## Additional Notes
- If you encounter a file creation error like `[WinError 183] Cannot create a file when that file already exists`, delete the existing output file before re-running processing.
- These steps were tested and confirmed to resolve the issue as of May 2025.

---

**This guide was created after resolving a real-world ffmpeg PATH issue in the facebeak project.** 