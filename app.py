"""Local-dev file server -- mirrors what Render's Static Site hosting serves
in production for this same folder. Not used in production."""

import os

from flask import Flask, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


@app.route("/")
@app.route("/<path:filename>")
def serve(filename=""):
    if filename == "":
        filename = "index.html"
    return send_from_directory(BASE_DIR, filename)


@app.route("/ping")
def ping():
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8130))
    print(f"Data Touchdown (local dev) running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
