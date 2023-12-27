import os
from datetime import datetime
import joblib
import pandas as pd
import warnings
from colorama import Fore
import json
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None

cols = ['PitcherTeam', 'Pitcher', 'PitcherThrows', 'TaggedPitchType', 'RelSpeed', 'SpinRate', 'SpinAxis',
        'InducedVertBreak', 'HorzBreak', 'HomeTeam', 'yt_Efficiency']
features = ['RelSpeed', 'SpinRate', 'VertBreak', 'HorzBreak', 'yt_Efficiency', 'SpinAxis']

pitch_name_mapping = {
    'Fastball': 0,
    'Sinker': 1,
    'Cutter': 2,
    'Splitter': 3,
    'Slider': 4,
    'Curveball': 5,
    'Changeup': 6,
}

reverse_map = {
    0: 'Fastball',
    1: 'Sinker',
    2: 'Cutter',
    3: 'Splitter',
    4: 'Slider',
    5: 'Curveball',
    6: 'Changeup'
}


def load_hashmap(filename):
    try:
        with open(filename, 'r') as file:
            hashmap = json.load(file)
    except FileNotFoundError:
        hashmap = {}
    return hashmap


pitcher_to_pitches = load_hashmap('pitcher_to_pitches.txt')

print(Fore.GREEN, "\nAUTOMATIC PITCH TAGGING CSV PROGRAM FOR JOLIET SLAMMERS")


def run_read_functions():
    input_csv = get_csv()
    model = load_model()
    df = predict_rows(model, input_csv)
    final = clean_rows(model, df)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_csv_path = "AutoTaggedCSVs/changeTeam_" + timestamp + ".csv"

    final.loc[final['PitcherThrows'] == 1, 'HorzBreak'] *= -1
    final.loc[final['PitcherThrows'] == 1, 'SpinAxis'] = 360 - final.loc[
        final['PitcherThrows'] == 1, 'SpinAxis']
    final['PitcherThrows'] = final['PitcherThrows'].replace({
        0: 'Right',
        1: 'Left'
    })

    final.to_csv(output_csv_path)
    print(final[['Pitcher', 'TaggedPitchType', 'AutoPitchType', 'RelSpeed', 'SpinRate']])


def run_add_functions():
    while True:
        pitcher_name = input("What is the full name of the pitcher you would like to add (or 'quit' to exit)? ")

        if pitcher_name.lower() == 'quit':
            break

        repertoire = []

        while True:
            pitch_type = input("Enter a pitch type or 'done' if finished adding pitches: ")

            if pitch_type.lower() == 'done':
                break

            repertoire.append(pitch_type)

        pitcher_to_pitches[pitcher_name] = repertoire

    for pitcher, repertoire in pitcher_to_pitches.items():
        print(f"{pitcher}: {', '.join(repertoire)}")
    save_hashmap('pitcher_to_pitches.txt', pitcher_to_pitches)


def save_hashmap(filename, hashmap):
    with open(filename, 'w') as file:
        json.dump(hashmap, file)


def ask_for_input():
    i = input("What would you like to do? \n(a) Add new pitcher\n(r) Read csv and generate auto tags\n")
    if i == 'a' or i == 'A':
        run_add_functions()
    elif i == 'r' or i == 'R':
        run_read_functions()
    else:
        print("Invalid input, try again")
        ask_for_input()


def get_csv():
    directory = input("Input the file path to the Yakkertech data: ")
    all_data = pd.DataFrame()

    # try:
    #     for filename in os.listdir(directory):
    #         if filename.endswith(".csv"):
    #             file_path = os.path.join(directory, filename)
    #             df = pd.read_csv(file_path)
    #             all_data = pd.concat([all_data, df], ignore_index=True)
    try:
        all_data = pd.read_csv(directory)
    except OSError:
        print("ERROR: File not found, try again")
        main()

    all_data['PitcherThrows'] = all_data['PitcherThrows'].replace({
        'Right': 0,
        'Left': 1
    })

    all_data.loc[all_data['PitcherThrows'] == 1, 'HorzBreak'] *= -1
    all_data.loc[all_data['PitcherThrows'] == 1, 'SpinAxis'] = 360 - all_data.loc[all_data['PitcherThrows'] == 1, 'SpinAxis']

    return all_data


def time_to_degrees(time_str):
    if isinstance(time_str, str):
        try:
            hh, mm = map(int, time_str.split(':'))
            return (hh * 360 / 12) + (mm * 360 / (12 * 60))
        except ValueError:
            pass
    return None


def convert_to_pname(pitch_type):
    if pitch_type is not None:
        pitch_name = reverse_map.get(pitch_type)

    else:
        pitch_name = None

    return pitch_name


def load_model():
    model = joblib.load('model/fl-pitch-tagging-model.joblib')
    return model


def predict_rows(model, csv):
    predictions = []
    for index, row in csv.iterrows():
        if not row[features].isnull().any():
            pred = model.predict(row[features].values.reshape(1, -1))
        else:
            pred = None
        if pred is not None:
            predictions.append(convert_to_pname(pred[0]))
        else:
            predictions.append(None)

    csv['AutoPitchType'] = predictions
    return csv


def clean_rows(model, csv):
    for index, row in csv.iterrows():
        pitcher = row['Pitcher']
        if pitcher in pitcher_to_pitches:
            repertoire = pitcher_to_pitches[pitcher]
            pred_probs = model.predict_proba(row[features].values.reshape(1, -1))[0]
            pitch_types = model.classes_

            pitch_probabilities = dict(zip(pitch_types, pred_probs))

            filtered_pitch_probabilities = {
                pitch_type: prob
                for pitch_type, prob in pitch_probabilities.items()
                if reverse_map[pitch_type] in repertoire
            }

            prediction = reverse_map[max(filtered_pitch_probabilities, key=filtered_pitch_probabilities.get)]
            csv.loc[index, 'AutoPitchType'] = prediction
    return csv


def main():
    ask_for_input()


if __name__ == "__main__":
    main()
