"""Weather via Open-Meteo (keyless, free).

Default location: Quezon City, Philippines. Override with "weather_city",
"weather_lat", "weather_lon" in config.json.

Open-Meteo (https://open-meteo.com) needs no API key, so the weather
command works out of the box on the Pi.
"""
import json
import logging
import urllib.parse
import urllib.request

log = logging.getLogger("weather")

DEFAULT_CITY = "Quezon City"
DEFAULT_LAT, DEFAULT_LON = 14.6357, 121.0325
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"


_WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    85: "Snow showers", 86: "Snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}


def _get_json(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "vcm/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _open_meteo(cfg, city, lat, lon, units):
    """Open-Meteo forecast API — no key required."""
    metric = units != "imperial"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                   "is_day,weather_code,wind_speed_10m",
        "hourly": "temperature_2m,weather_code",
        "daily": "sunrise,sunset",
        "forecast_days": 2,
        "timezone": "auto",
        "wind_speed_unit": "kmh" if metric else "mph",
        "temperature_unit": "celsius" if metric else "fahrenheit",
    }
    data = _get_json(OPEN_METEO_BASE + "?" + urllib.parse.urlencode(params))
    cur = data.get("current", {})
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})
    times = hourly.get("time", [])[:12]
    now_t = (cur.get("time") or "")[:13]  # "YYYY-MM-DDTHH"
    start = 0
    for i, t in enumerate(times):
        if t[:13] >= now_t:
            start = i
            break
    idx = list(range(start, min(start + 12, len(times))))
    return {
        "ok": True,
        "city": city,
        "temp_c": round(float(cur.get("temperature_2m", 0.0)), 1),
        "feels_like_c": round(float(cur.get("apparent_temperature", 0.0)), 1),
        "humidity": cur.get("relative_humidity_2m"),
        "wind_kmh": round(float(cur.get("wind_speed_10m", 0.0)), 1),
        "condition": _WMO.get(int(cur.get("weather_code") if cur.get("weather_code") is not None else -1), "Unknown"),
        "sunrise": (daily.get("sunrise") or [None])[0],
        "sunset": (daily.get("sunset") or [None])[0],
        "hourly": [
            {"time": times[i], "temp_c": round(float(
                hourly.get("temperature_2m", [0] * len(times))[i]), 1),
             "condition": _WMO.get(int(hourly.get(
                 "weather_code", [-1] * len(times))[i]), "Unknown")}
            for i in idx
        ],
        "provider": "open-meteo",
    }


def get_weather(cfg):
    """Return a dict suitable for the UI / spoken response:
    {ok, city, temp_c, feels_like_c, humidity, wind_kmh, condition,
     sunrise, sunset, hourly:[{time,temp_c,condition}], provider, error}"""
    city = cfg.get("weather_city", DEFAULT_CITY)
    lat = cfg.get("weather_lat", DEFAULT_LAT)
    lon = cfg.get("weather_lon", DEFAULT_LON)
    units = cfg.get("weather_units", "metric")
    try:
        return _open_meteo(cfg, city, lat, lon, units)
    except Exception as e:
        log.warning("open-meteo request failed: %s", e)
        return {"ok": False, "city": city, "error": str(e)}