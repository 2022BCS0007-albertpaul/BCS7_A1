from datetime import datetime, timedelta

def extract_features(customer):
    tickets = customer["tickets"]
    now = datetime.now()

    def count_days(days):
        return sum(
            1 for t in tickets
            if datetime.fromisoformat(t["date"]) > now - timedelta(days=days)
        )

    # Ticket frequency
    freq_7d = count_days(7)
    freq_30d = count_days(30)
    freq_90d = count_days(90)

    # Complaint count
    complaint_count = sum(1 for t in tickets if t["type"] == "complaint")

    # Avg time between tickets
    dates = sorted([datetime.fromisoformat(t["date"]) for t in tickets])
    if len(dates) > 1:
        gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        avg_gap = sum(gaps) / len(gaps)
    else:
        avg_gap = 0

    # Charge change
    charge_diff = customer["monthly_charges"] - customer["previous_month_charges"]

    return [
        freq_7d,
        freq_30d,
        freq_90d,
        complaint_count,
        avg_gap,
        charge_diff
    ]