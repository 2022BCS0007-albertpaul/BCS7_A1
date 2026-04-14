from datetime import datetime, timedelta

def calculate_risk(data):
    tickets = data["tickets"]
    now = datetime.now()

    recent_tickets = [
        t for t in tickets
        if datetime.fromisoformat(t["date"]) > now - timedelta(days=30)
    ]

    monthly = data["monthly_charges"]
    previous = data["previous_month_charges"]
    contract = data["contract_type"]

    # -------------------
    # HIGH RISK RULES
    # -------------------

    # Many recent complaints
    if len(recent_tickets) > 5:
        return "HIGH"

    # Any complaint + month-to-month contract
    if contract == "Month-to-Month" and any(t["type"] == "complaint" for t in tickets):
        return "HIGH"

    # Extreme charge increase (IMPORTANT for your new_data.json)
    if previous > 0 and (monthly - previous) / previous >= 1.0:  # ≥100% increase
        return "HIGH"

    # -------------------
    # MEDIUM RISK RULES
    # -------------------

    if monthly > previous and len(tickets) >= 3:
        return "MEDIUM"

    # -------------------
    # LOW RISK
    # -------------------
    return "LOW"