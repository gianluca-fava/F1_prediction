import fastf1
import pandas as pd
import os
import logging

# --- CONFIGURATION ---
CACHE_DIR = './f1_cache'
BASE_OUTPUT_DIR = 'data'
YEARS = [2025]
# Added 'Q' for Qualifying and 'R' for Race
SESSIONS = ['Q', 'R']

fastf1.Cache.enable_cache(CACHE_DIR)
logging.basicConfig(level=logging.INFO)


def download_session_data(year, event_name, round_num, session_type):
    """
    Downloads a specific session (Q or R) and saves it to its own parquet file.
    Overwrites existing files to ensure a fresh download.
    """
    year_dir = os.path.join(BASE_OUTPUT_DIR, str(year))
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)

    # File name includes session type to avoid overwriting R with Q or viceversa
    file_path = os.path.join(year_dir, f"gp_{round_num:02d}_{session_type}.parquet")

    # NOTE: os.path.exists check removed to force redownload as requested

    try:
        print(f"--- Downloading: {year} Round {round_num} - {event_name} [{session_type}] ---")
        session = fastf1.get_session(year, event_name, session_type)

        # Attempt full load
        try:
            session.load(laps=True, telemetry=False, weather=True)
        except Exception as e:
            print(f"      [!] Weather failed for {session_type}, retrying laps only...")
            session.load(laps=True, telemetry=False, weather=False)

        if session.laps is not None and not session.laps.empty:
            df = session.laps.copy()

            # Enrich with Metadata
            df['EventName'] = event_name
            df['Year'] = year
            df['RoundNumber'] = round_num
            df['SessionType'] = session_type  # Essential to distinguish data later

            # Weather extraction logic
            if not session.weather_data.empty:
                df['AvgTrackTemp'] = session.weather_data['TrackTemp'].mean()
                df['AvgHumidity'] = session.weather_data['Humidity'].mean()
            else:
                df['AvgTrackTemp'] = df['AvgHumidity'] = None

            # Save to Parquet
            df.to_parquet(file_path, engine='pyarrow', index=False)
            print(f"      [OK] Saved to {file_path}")
            return True
        else:
            print(f"      [EMPTY] No laps found for {session_type}.")
            return False

    except Exception as e:
        print(f"      [ERROR] {event_name} [{session_type}] failed: {e}")
        return False


def main():
    for year in YEARS:
        print(f"\n\n>>> STARTING SEASON {year} (Full Re-download)")
        schedule = fastf1.get_event_schedule(year)
        race_events = schedule[schedule['EventFormat'] != 'testing']

        failed_sessions = []

        for _, event in race_events.iterrows():
            for sess in SESSIONS:
                success = download_session_data(year, event['EventName'], event['RoundNumber'], sess)
                if not success:
                    failed_sessions.append(f"{event['EventName']} [{sess}]")

        if failed_sessions:
            print(f"\nSeason {year} finished with errors in: {failed_sessions}")
        else:
            print(f"\nSeason {year} completed successfully!")


if __name__ == "__main__":
    main()