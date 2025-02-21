# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import joblib
import os
import json

app = Flask(__name__)

# Load Sentiment Model
MODEL_PATH = "model/nb_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise Exception("❌ Model file not found. Train the model first.")

# Mock database for food order tracking
ORDERS_FILE = "orders.json"

# Ensure the orders file exists
if not os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "w") as f:
        json.dump({}, f)

# Load orders
def load_orders():
    with open(ORDERS_FILE, "r") as f:
        return json.load(f)

# Save orders
def save_orders(orders):
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=4)

@app.route("/")
def home():
    return "✅ Food Order Tracking & Sentiment Analysis API Running!"

# 📌 1️⃣ Place New Order
@app.route("/place_order", methods=["POST"])
def place_order():
    data = request.json
    order_id = str(len(load_orders()) + 1)  # Auto-increment order ID
    order_data = {
        "order_id": order_id,
        "customer_name": data.get("customer_name"),
        "food_item": data.get("food_item"),
        "status": "Preparing"
    }

    orders = load_orders()
    orders[order_id] = order_data
    save_orders(orders)

    return jsonify({"message": "✅ Order placed successfully!", "order": order_data})

# 📌 2️⃣ Check Order Status
@app.route("/order_status/<order_id>", methods=["GET"])
def order_status(order_id):
    orders = load_orders()
    if order_id in orders:
        return jsonify({"order_status": orders[order_id]["status"]})
    return jsonify({"error": "❌ Order not found"}), 404

# 📌 3️⃣ Update Order Status
@app.route("/update_status", methods=["POST"])
def update_status():
    data = request.json
    order_id = data.get("order_id")
    new_status = data.get("status")

    orders = load_orders()
    if order_id in orders:
        orders[order_id]["status"] = new_status
        save_orders(orders)
        return jsonify({"message": "✅ Order status updated successfully!"})
    return jsonify({"error": "❌ Order not found"}), 404

# 📌 4️⃣ Analyze Customer Feedback
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "❌ No feedback text provided"}), 400

    prediction = model.predict([text])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({"feedback": text, "sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)

