""" This script hosts the documentation on a local server. The documentation is stored in a
    repository and is updated when the server is started, or when a user requests an update.

    This script is automatically run when the Docker container (also in this folder) is started.

    >> docker build -t docs_server . (if you haven't built the image yet)
    >> docker run -d -p 6001:6001 --name docs_server --restart unless-stopped docs_server
"""

import os
import subprocess

from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

# Define the path to the docs directory
DOCS_DIR = "/app/repo/docs/usbmd"


def update_repo():

    token = "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"
    repo_url = f"https://{token}@github.com/tue-bmd/ultrasound-toolbox.git"
    repo_dir = "/app/repo"

    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    subprocess.run(["git", "-C", repo_dir, "checkout", "main"], check=True)
    subprocess.run(["git", "-C", repo_dir, "pull", "origin", "main"], check=True)


@app.route("/update")
def update_docs():
    try:
        update_repo()
        return jsonify({"message": "Repository updated"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/<path:path>")
def serve_docs(path):
    try:
        update_repo()
        return send_from_directory(DOCS_DIR, path)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    try:
        update_repo()
        return send_from_directory(DOCS_DIR, "index.html")
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6001)
