from flask import Flask, request, jsonify
from utils.bulasik_utils import run_bulasik_advice
from utils.camasir_utils import run_camasir_advice
from utils.buzdolabi_utils import run_buzdolabi_advice
from utils.tv_utils import run_tv_advice

app = Flask(__name__)

DEVICE_FUNCTIONS = {
    "bulasik": run_bulasik_advice,
    "camasir": run_camasir_advice,
    "buzdolabi": run_buzdolabi_advice,
    "tv": run_tv_advice,
}

@app.route("/advice", methods=["GET"])
def get_advice():
    device_type = request.args.get("device_type")

    if not device_type:
        return jsonify({"error": "device_type parametresi eksik"}), 400

    if device_type not in DEVICE_FUNCTIONS:
        return jsonify({"error": f"Desteklenmeyen cihaz: {device_type}"}), 400

    try:
        result = DEVICE_FUNCTIONS[device_type]()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"{device_type} için tahmin sırasında hata oluştu", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
