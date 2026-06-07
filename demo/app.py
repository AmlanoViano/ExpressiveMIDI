from flask import Flask, request, send_file, render_template, jsonify
import os, glob, tempfile
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.humanise import humanise

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.mkdtemp()

def get_latest_model(pattern):
    models = sorted(glob.glob(f"experiments/{pattern}"))
    return models[-1] if models else None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/humanise", methods=["POST"])
def humanise_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f           = request.files["file"]
    instrument  = request.form.get("instrument", "piano")
    strength    = float(request.form.get("strength", 1.0))
    use_expr    = request.form.get("expression", "true") == "true"

    input_path  = os.path.join(UPLOAD_FOLDER, "input.mid")
    output_path = os.path.join(UPLOAD_FOLDER, "output.mid")
    f.save(input_path)

    if instrument == "strings":
        timing_model = get_latest_model("best_strings_*.pt")
        expr_model   = None
    else:
        timing_model = get_latest_model("best_hybrid_*.pt")
        expr_model   = get_latest_model("best_expression_*.pt") if use_expr else None

    if not timing_model:
        return jsonify({"error": "No trained model found"}), 500

    try:
        humanise(input_path, output_path, timing_model, expr_model, strength)
        return send_file(output_path, as_attachment=True,
                        download_name="humanised.mid",
                        mimetype="audio/midi")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5050)
