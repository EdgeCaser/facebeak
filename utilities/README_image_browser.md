# Kivy Image Browser for SSH

A Kivy-based image browser designed for viewing crop pictures remotely via SSH with X11 forwarding.

## Features

- **Folder Navigation**: Browse directories with a file tree on the left
- **Image Viewing**: Display images with proper scaling and aspect ratio
- **Navigation Controls**: Previous/Next buttons and keyboard navigation
- **File Information**: Shows image count, filename, and file size
- **Multiple Formats**: Supports JPG, PNG, BMP, GIF, TIFF, WebP
- **Background Loading**: Threaded image loading for smooth performance
- **SSH Friendly**: Optimized for remote viewing over SSH

## Installation

1. Install dependencies:
```bash
pip install -r requirements_image_browser.txt
```

2. For SSH usage, ensure X11 forwarding is enabled:
```bash
ssh -X username@hostname
# or
ssh -Y username@hostname  # for trusted X11 forwarding
```

## Usage

### Basic Usage
```bash
python kivy_image_browser.py
```

### Start in Specific Directory
```bash
python kivy_image_browser.py /path/to/your/images
```

### For Crow Crops
```bash
python kivy_image_browser.py /home/ubuntu/facebeak/crow_crops
```

## Controls

### Mouse Controls
- **Click directory**: Navigate into folder
- **Click image file**: View image
- **Click "Up" button**: Go to parent directory
- **Click "Refresh" button**: Reload current directory
- **Click "Previous/Next"**: Navigate between images

### Keyboard Shortcuts
- **Left Arrow**: Previous image
- **Right Arrow**: Next image
- **Escape**: Close application

## Interface Layout

```
┌─────────────────┬─────────────────────────────────┐
│   File Browser  │        Image Viewer             │
│                 │                                 │
│ Path: /path/... │ [Image Info] [Prev] [Next]     │
│ ┌─────────────┐ │                                 │
│ │ folder1/    │ │                                 │
│ │ folder2/    │ │         [Image Display]         │
│ │ image1.jpg  │ │                                 │
│ │ image2.png  │ │                                 │
│ │ ...         │ │                                 │
│ └─────────────┘ │                                 │
└─────────────────┴─────────────────────────────────┘
```

## SSH Setup Tips

### Enable X11 Forwarding on Server
Edit `/etc/ssh/sshd_config`:
```
X11Forwarding yes
X11DisplayOffset 10
X11UseLocalhost yes
```

### Connect with X11 Forwarding
```bash
ssh -X user@server
# Test X11 forwarding works:
xeyes  # Should show a window with eyes
```

### Performance Tips for SSH
- Use compression: `ssh -X -C user@server`
- For slow connections, consider using VNC instead
- Ensure your local X server is running (automatic on most Linux desktops)

## Troubleshooting

### "Cannot connect to display" Error
- Ensure X11 forwarding is enabled: `echo $DISPLAY`
- Try: `export DISPLAY=:0.0`
- Check SSH connection: `ssh -X -v user@server`

### Kivy Installation Issues
```bash
# On Ubuntu/Debian:
sudo apt-get install python3-kivy

# Or via pip with system dependencies:
sudo apt-get install python3-dev python3-pip build-essential
pip install kivy[base]
```

### Performance Issues
- Reduce image quality for faster loading over SSH
- Use local file system instead of network mounts when possible
- Consider using the Flask browser (`browse_ec2_images.py`) for web-based viewing

## Alternative: Web Browser

For situations where X11 forwarding is not available, use the Flask-based browser:
```bash
python browse_ec2_images.py
# Then access via browser at http://server-ip:8080
``` 