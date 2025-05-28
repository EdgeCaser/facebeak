# save as browse_images.py
from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)
IMAGE_DIR = '/home/ubuntu/facebeak/crow_crops'

@app.route('/')
def index():
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = sorted(files)[:500]  # paginate or lazy load as needed
    html = '<html><body>{}</body></html>'.format(
        ''.join(f'<img src="/img/{f}" width="200" style="margin:5px">' for f in files))
    return render_template_string(html)

@app.route('/img/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
