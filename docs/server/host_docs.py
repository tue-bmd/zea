""" This script hosts the documentation on a local server. The documentation is stored in a
    repository and is updated when the server is started, or when a user requests an update.

    This script is automatically run when the Docker container (also in this folder) is started.

    1). Set the environment variables in the default.env file.

    2). Create a Docker volume to store the repository:
        >> docker volume create ultrasound-toolbox-repo

    3). Build the Docker image:
        >> docker build -t docs_server . (if you haven't built the image yet)

    4). Run the Docker container:
        >> docker run -d --env-file default.env -v /var/run/docker.sock:/var/run/docker.sock -v
        ultrasound-toolbox-repo:/app/repo -p 6001:6001 --name docs_server
        --restart unless-stopped docs_server


"""

import os
import subprocess
import threading
from functools import wraps

from authlib.integrations.flask_client import OAuth
from flask import Flask, jsonify, redirect, send_from_directory, session, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default_secret_key")
app.config["GITHUB_CLIENT_ID"] = os.environ.get("GITHUB_CLIENT_ID")
app.config["GITHUB_CLIENT_SECRET"] = os.environ.get("GITHUB_CLIENT_SECRET")

oauth = OAuth(app)

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


## Helper functions
class DocUpdater:
    """Class to handle the update process for the documentation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._update_thread = None

    def start_update(self):
        """Start the update process in a background thread,
        ensuring that only one update runs at a time."""
        with self._lock:
            if self._update_thread and self._update_thread.is_alive():
                print(
                    "An update is already in progress. Please wait until it completes."
                )
                return "Update in progress"

            self._update_thread = threading.Thread(target=self._update_process)
            self._update_thread.start()
            return "Update started"

    def update_repo(self):
        """Update the repository with the latest changes from the remote repository."""
        GITHUB_PAT = os.environ.get("GITHUB_PAT")
        repo_url = f"https://{GITHUB_PAT}@github.com/tue-bmd/ultrasound-toolbox.git"
        repo_dir = "/app/repo"

        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", repo_dir],
            check=True,
        )

        if not os.path.exists(repo_dir):
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

        subprocess.run(["git", "-C", repo_dir, "checkout", "main"], check=True)
        subprocess.run(["git", "-C", repo_dir, "pull", "origin", "main"], check=True)

    def build_docker_image(self):
        """Build the Docker image from the Dockerfile in the repository."""
        repo_dir = "/app/repo"
        image_name = "usbmd"

        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                image_name,
                repo_dir,
                "--build-arg",
                "KERAS3=True",
            ],
            check=True,
        )

    def run_pdoc_inside_docker(self):
        """Run pdoc to generate the documentation inside the Docker container."""
        repo_dir = "/app/repo"
        volume_name = "ultrasound-toolbox-repo"

        # Ensure the Docker volume is created (if not already done)
        subprocess.run(["docker", "volume", "create", volume_name], check=True)

        # Remove old HTML files
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{volume_name}:{repo_dir}",
                "busybox",
                "sh",
                "-c",
                f"find {repo_dir}/docs/usbmd/ -name '*.html' -type f -delete",
            ],
            check=True,
        )

        # Run pdoc inside the Docker container
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{volume_name}:{repo_dir}",
                "-w",
                repo_dir,
                "usbmd:latest",
                "sh",
                "-c",
                (
                    "pip install pdoc3 && pdoc usbmd --html --output-dir /app/repo/docs "
                    "--force --skip-errors --template-dir /app/repo/docs/pdoc_template"
                ),
            ],
            check=True,
        )

    def _update_process(self):
        """Internal method to handle the update process."""
        try:
            print("Starting the update process...")
            self.update_repo()
            self.build_docker_image()
            self.run_pdoc_inside_docker()
            print("Update process completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during the subprocess execution: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during the update process: {e}")
        finally:
            self._lock.release()

    def busy(self):
        """Check if an update is in progress."""
        return self._lock.locked()


## Routes
doc_updater = DocUpdater()


def login_required(f):
    """Decorator to require login."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "github_token" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


@app.route("/update")
# @login_required
def update_docs():
    """Manually force an update of the repository."""
    try:
        message = doc_updater.start_update()
        return jsonify({"message": message}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500


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


@app.route("/<path:path>")
# @login_required
def serve_docs(path):
    """Serve the documentation."""
    try:
        return send_from_directory(DOCS_DIR, path)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
# @login_required
def index():
    """Serve the index page."""
    try:
        return send_from_directory(DOCS_DIR, "index.html")
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6001)
