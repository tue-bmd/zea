""" This script hosts the documentation on a local server. The documentation is stored in a
    repository and is updated when the server is started, or when a user requests an update.

    This script is automatically run when the Docker container (also in this folder) is started.

    >> docker build -t docs_server . (if you haven't built the image yet)
    >> docker run -d -p 6001:6001 --name docs_server --restart unless-stopped docs_server
"""

import os
import subprocess
from functools import wraps

from authlib.integrations.flask_client import OAuth
from flask import Flask, jsonify, redirect, send_from_directory, session, url_for

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", "default_secret_key"
)  # Use a real secret key in production
# app.secret_key = 'your_secret_key'  # Replace with a real secret key
oauth = OAuth(app)

# Configure GitHub OAuth client

# TODO: These should be environment variables, not hardcoded in the script. (REGENERATE)
app.config["GITHUB_CLIENT_ID"] = "Ov23lixwYEES6JAk6Moj"
app.config["GITHUB_CLIENT_SECRET"] = "1846fcb16eb048d0d2e1145c1a57dcc4978f0f3b"

github = oauth.register(
    name="github",
    client_id=app.config["GITHUB_CLIENT_ID"],
    client_secret=app.config["GITHUB_CLIENT_SECRET"],
    authorize_url="https://github.com/login/oauth/authorize",
    authorize_params=None,
    access_token_url="https://github.com/login/oauth/access_token",
    access_token_params=None,
    client_kwargs={"scope": "user:email"},
)

DOCS_DIR = "/app/repo/docs/usbmd"
# This is a personal access token (PAT) for the GitHub repository, owned by the BMD group, with
# read-only acces to the repository. Not the most secure way to do this, but since it is
# a read-only token, it is not a big security risk.
# TODO: These should be environment variables, not hardcoded in the script. (REGENERATE)
TOKEN = """github_pat_\
    11ANWVESA0sJShCnhONCjG_kKdrvvHjF35AX9KkSSVmYCYvGDKaD0L0K1sV7qs6WJjH44Z2EIP1UhzhzTB"""


def update_repo():
    """Update the repository with the latest changes from the remote repository."""
    repo_url = f"https://{TOKEN}@github.com/tue-bmd/ultrasound-toolbox.git"
    repo_dir = "/app/repo"

    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    subprocess.run(["git", "-C", repo_dir, "checkout", "main"], check=True)
    subprocess.run(["git", "-C", repo_dir, "pull", "origin", "main"], check=True)


def login_required(f):
    """Decorator to require login."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "github_token" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


@app.route("/login")
def login():
    """Login with GitHub."""
    redirect_uri = url_for("authorize", _external=True)
    return github.authorize_redirect(redirect_uri)


@app.route("/authorize")
def authorize():
    """Authorize the user with GitHub."""
    token = github.authorize_access_token()
    user_info = github.get("https://api.github.com/user", token=token).json()
    session["github_token"] = token
    session["github_user"] = user_info["login"]
    return redirect(url_for("index"))


@app.route("/update")
@login_required
def update_docs():
    """Manually force an update of the repository."""
    try:
        update_repo()
        return jsonify({"message": "Repository updated"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/<path:path>")
@login_required
def serve_docs(path):
    """Serve the documentation."""
    try:
        update_repo()
        return send_from_directory(DOCS_DIR, path)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
@login_required
def index():
    """Serve the index page."""
    try:
        update_repo()
        return send_from_directory(DOCS_DIR, "index.html")
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6001)
