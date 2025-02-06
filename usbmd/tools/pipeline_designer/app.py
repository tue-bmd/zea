"""This module contains a web-based GUI for designing and checking pipelines."""

import inspect

import yaml
from flask import Flask, jsonify, request, send_from_directory

import usbmd.ops_v2 as ops_v2
from usbmd.registry import ops_registry

app = Flask(__name__, static_folder="static")


@app.route("/")
def serve_frontend():
    """Serve the main HTML file."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/operations", methods=["GET"])
def get_operations():
    """Fetch the list of registered operations."""
    operations = list(ops_registry.registered_names())

    parsed_ops = []

    for op in operations:
        op_class = ops_registry[op]

        # Get the input and output types of the operation
        init_keys = set(inspect.signature(op_class.__init__).parameters.keys())

        # strip self and kwargs
        init_keys.discard("self")
        init_keys.discard("kwargs")

        # check if multiple inputs are allowed
        allow_multiple_inputs = True if op in ops_v2.MULTIPLE_INPUT_OPS else False

        parsed_ops.append(
            {
                "name": op,
                "init_keys": list(init_keys),
                "allow_multiple_inputs": allow_multiple_inputs,
            }
        )
    return jsonify(parsed_ops)


@app.route("/save_pipeline", methods=["POST"])
def save_pipeline():
    """Save the pipeline configuration to a YAML file."""
    pipeline = request.json
    with open("pipeline_config.yaml", "w") as file:
        yaml.dump(pipeline, file)
    return jsonify({"status": "success", "message": "Pipeline saved successfully!"})


if __name__ == "__main__":
    app.run(debug=True)
