from datetime import datetime


def extract_time_features(ms_timestamp: int):
    dt = datetime.fromtimestamp(ms_timestamp / 1000)

    features = {
        "timestamp": dt.isoformat(),
        "date": dt.strftime("%Y-%m-%d"),
        "day": dt.strftime("%A"),
        "hour": dt.hour,
        "minute": dt.minute,
        "is_weekend": dt.weekday() >= 5,
        "time_of_day": get_time_of_day(dt.hour),
    }

    return features


def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"
