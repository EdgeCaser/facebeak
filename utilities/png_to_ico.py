from PIL import Image
import sys
import os

def convert_png_to_ico(png_path):
    if not os.path.isfile(png_path):
        print(f"File not found: {png_path}")
        return

    base_name = os.path.splitext(png_path)[0]
    ico_path = f"{base_name}.ico"

    img = Image.open(png_path).convert("RGBA")
    img.save(ico_path, format="ICO", sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])

    print(f"Saved: {ico_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python png_to_ico.py path/to/image.png")
    else:
        convert_png_to_ico(sys.argv[1])
