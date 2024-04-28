import datetime
import os
from flask import Flask, jsonify, render_template, request, session
from faith.library.string_library import StringLibrary
from faith.library.utils import get_config
from faith.library.temporal_annotator.sutime_date_annotator import SutimeAnnotator

"""Flask config"""
app = Flask(__name__)
# Set the secret key to some random bytes.
app.secret_key = os.urandom(32)
app.permanent_session_lifetime = datetime.timedelta(days=365)

"""Load modules"""
config_file = "faith/sutime_service/sutime.yml"
config = get_config(config_file)
string_lib = StringLibrary(config)
sutime = SutimeAnnotator(config, string_lib)

"""Routes"""
@app.route("/test", methods=["GET"])
def test():
    return "Test successful!"

@app.route("/multithread", methods=["POST"])
def multithread():
    json_dict = request.json
    string_refers = json_dict.get("string_refers")
    annotate_strings = sutime.sutime_annotation_normalization_multithreading(string_refers)
    return jsonify(annotate_strings)

@app.route("/annotation", methods=["POST"])
def annotation():
    json_dict = request.json
    string = json_dict.get("string")
    reference_time = json_dict.get("reference_time")
    annotate_strings = sutime.sutime_annotation_normalization(string, reference_time)
    return jsonify(annotate_strings)

if __name__ == "__main__":
    app.run(host="localhost", port=7779, threaded=True)