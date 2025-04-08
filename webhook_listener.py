from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

# Webhook route to handle incoming webhook notifications from Strava
@app.route("/strava-webhook", methods=["POST", "GET"])
def strava_webhook():
    if request.method == "GET":
        # Webhook verification process (when Strava initially registers the webhook)
        verify_token = request.args.get('hub.challenge')
        if verify_token:
            return verify_token  # Respond with the verification token to confirm the webhook
        return "Invalid request", 400

    if request.method == "POST":
        # Handle the POST request sent by Strava when a relevant event happens
        event_data = request.json
        # Log the incoming event for debugging
        print(json.dumps(event_data, indent=4))

        # Process the event (for example, check for new activity)
        if "object_type" in event_data and event_data["object_type"] == "activity":
            activity_id = event_data["object_id"]
            print(f"New activity uploaded with ID: {activity_id}")
            
            # Here, you would typically fetch the activity data from Strava API
            # Example: fetch_activity_data(activity_id)

            # Optionally, store the activity ID or update the database
            # Example: save_activity_to_db(activity_id)

        return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(port=5000)
