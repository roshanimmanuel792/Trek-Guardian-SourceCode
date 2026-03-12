from flask import Flask, request, jsonify
import datetime

app = Flask(__name__)

@app.route("/alert", methods=["POST"])
def receive_alert():

    data = request.json

    spo2 = data.get("spo2")
    altitude = data.get("altitude")
    heartRate = data.get("heartRate")
    latitude = data.get("lat")
    longitude = data.get("lon")
    risk = data.get("risk")

    timestamp = datetime.datetime.now()

    print("----- EMERGENCY ALERT RECEIVED -----")
    print("Time:", timestamp)
    print("SpO2:", spo2)
    print("Altitude:", altitude)
    print("Heart Rate:", heartRate)
    print("Location:", latitude, longitude)
    print("Risk Level:", risk)
    print("-----------------------------------")

    return jsonify({"status": "alert received"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
