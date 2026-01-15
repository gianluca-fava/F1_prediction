import fastf1
import pandas as pd
import os
import logging

# --- CONFIGURATION ---
CACHE_DIR = './f1_cache'
OUTPUT_DIR = 'data/full_data'
# For Strategy Prediction, 2018-2025 is the "Gold Standard" for data quality
START_YEAR = 2018
END_YEAR = 2025
YEARS_TO_DOWNLOAD = list(range(START_YEAR, END_YEAR + 1))

# Create necessary directories
for folder in [CACHE_DIR, OUTPUT_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

# Setup FastF1 cache and logging
fastf1.Cache.enable_cache(CACHE_DIR)
logging.basicConfig(level=logging.INFO)


def fetch_season_data(year):
    """
    Downloads and processes all race sessions for a given year using the working logic.
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        # Filter out testing events
        race_events = schedule[schedule['EventFormat'] != 'testing']
    except Exception as e:
        print(f"Error fetching schedule for {year}: {e}")
        return None

    all_laps_list = []

    for _, event in race_events.iterrows():
        event_name = event['EventName']
        round_num = event['RoundNumber']

        try:
            print(f"--- Fetching: {year} Round {round_num} - {event_name} ---")

            # Load race session ('R')
            session = fastf1.get_session(year, event_name, 'R')
            session.load(laps=True, telemetry=False, weather=True)

            # Extract laps dataframe
            laps = session.laps.copy()

            # Process weather data (get average conditions for the race)
            weather = session.weather_data
            if not weather.empty:
                avg_track_temp = weather['TrackTemp'].mean()
                avg_air_temp = weather['AirTemp'].mean()
                avg_humidity = weather['Humidity'].mean()
            else:
                avg_track_temp = avg_air_temp = avg_humidity = None

            # Add metadata and weather features
            laps['EventName'] = event_name
            laps['Year'] = year
            laps['RoundNumber'] = round_num
            laps['AvgTrackTemp'] = avg_track_temp
            laps['AvgAirTemp'] = avg_air_temp
            laps['AvgHumidity'] = avg_humidity

            all_laps_list.append(laps)
            print(f"Successfully processed {event_name}")

        except Exception as e:
            print(f"Error skipping {event_name}: {e}")
            continue

    if all_laps_list:
        return pd.concat(all_laps_list, ignore_index=True)
    return None


def main():
    print(f"Starting data ingestion for years: {YEARS_TO_DOWNLOAD}")

    for year in YEARS_TO_DOWNLOAD:
        output_file = f"{OUTPUT_DIR}/raw_f1_{year}.parquet"

        # Skip if already downloaded to save time
        if os.path.exists(output_file):
            print(f"Year {year} already exists. Skipping...")
            continue

        dataset = fetch_season_data(year)

        if dataset is not None:
            # Using pyarrow engine for high-performance Parquet writing
            dataset.to_parquet(output_file, engine='pyarrow', index=False)
            print(f"SUCCESS: {year} saved. Rows: {len(dataset)}")
        else:
            print(f"No data was collected for {year}.")


if __name__ == "__main__":
    main()