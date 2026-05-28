import requests
import pandas as pd
from datetime import datetime

# Target location
latitude = 35.0    # °N
longitude = 104.0  # °E

# Query date range
start_date = "20230101"
end_date = "20231231"

# ALLSKY_SFC_SW_DWN = surface solar irradiance (W/m^2), also called GHI
# T2M = air temperature at 2 metres (°C)
parameters = "ALLSKY_SFC_SW_DWN,T2M"


def fetch_data(lat, lon, start, end, params):
    """Download hourly meteorological data from NASA POWER."""

    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters={params}"
        "&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start}"
        f"&end={end}"
        "&format=JSON"
    )

    print("Downloading data from NASA POWER...")
    print(f"  Location: ({lat}, {lon})")
    print(f"  Period: {start} ~ {end}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    raw_data = response.json()

    # Parse into DataFrame
    records = raw_data["properties"]["parameter"]
    df = pd.DataFrame(records)

    # Convert index to datetime
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df.index.name = "datetime"

    df.rename(columns={
        "ALLSKY_SFC_SW_DWN": "GHI",
        "T2M": "T_amb"
    }, inplace=True)

    df["GHI"] = df["GHI"].clip(lower=0)

    annual_ghi = df["GHI"].mean()
    annual_temp = df["T_amb"].mean()

    print(f"  Done. Retrieved {len(df)} hourly records")
    print(f"  Annual mean GHI: {annual_ghi:.1f} W/m^2")
    print(f"  Annual mean temperature: {annual_temp:.1f} °C")

    return df


def save_data(data_df, path):
    """Save DataFrame to CSV."""
    data_df.to_csv(path)
    print(f"  Saved to: {path}")


if __name__ == "__main__":
    df = fetch_data(latitude, longitude, start_date, end_date, parameters)
    save_data(df, "data/nasa_power_data.csv")

    # Quick preview
    print("\n--- Preview ---")
    print(df.head(10))
