import fetch
import analyze_data
import pandas as pd

activity_ids = []

def main():
    print("Welcome to Strava Analysis!")
    df = analyze_data.import_data()

    full_id_list = analyze_data.get_ids(df)

    analyze_data.map_activity(full_id_list[0])
        



if __name__ == "__main__":
    main()