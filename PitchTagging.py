import pandas as pd
import numpy as np
from pybaseball import statcast, cache
from sklearn.model_selection import train_test_split
import warnings
from colorama import Fore
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

cache.enable()
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None

cols = ['release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z', 'p_throws', 'pitch_name', 'ax', 'ay', 'az', 'vx0', 'vy0',
        'vz0']
features = ['release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z', 'p_throws', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']

pitch_types_sc = ['4-Seam Fastball', 'Sinker', 'Cutter', 'Slider', 'Changeup', 'Curveball', 'Split-Finger']
pitch_types_yakker = ['Fastball', 'Sinker', 'Cutter', 'Slider', 'Changeup', 'Curveball', 'Splitter']

pitcher_to_pitches = {
    'Cam Aufderheide': ['Fastball', 'Sinker', 'Cutter', 'Curveball'],
    'Chandler Brierley': ['Sinker', 'Slider', 'Splitter'],
    'Cole Cook': ['Fastball', 'Slider', 'Cutter', 'Curveball', 'Sinker', 'Changeup'],
    'David Harrison': ['Sinker', 'Changeup', 'Slider'],
    'Tyler Jay': ['Fastball', 'Sinker', 'Slider', 'Curveball', 'Changeup'],
    'Tanner Kiest': ['Fastball', 'Slider', 'Changeup'],
    'Tanner Larkins': ['Fastball', 'Sinker', 'Slider', 'Curveball', 'Changeup'],
    'Jared Liebelt': ['Sinker', 'Slider', 'Changeup'],
    'Will MacLean': ['Sinker', 'Splitter', 'Slider', 'Curveball'],
    'Caden O\'brien': ['Fastball', 'Changeup', 'Slider'],
    'Evy Ruibal': ['Fastball', 'Slider', 'Curveball', 'Sinker'],
    'Cole Stanton': ['Fastball', 'Curveball', 'Changeup', 'Cutter'],
    'Brad VanAsdlen': ['Fastball', 'Sinker', 'Splitter', 'Cutter'],
}

print(Fore.GREEN, "\nAUTOMATIC PITCH TAGGING CSV PROGRAM FOR JOLIET SLAMMERS")


def run_read_functions():
    input_csv, string = get_csv()
    X, y = load_data()
    model = train_model(X, y)
    df = predict_rows(model, input_csv)
    final = adjust_changeups(clean_rows(model, df))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    output_csv_path = "AutoTaggedCSVs/autoTagged_" + string
    final.to_csv(output_csv_path)
    print(final[['Pitcher', 'TaggedPitchType', 'AutoPitchType', 'release_speed']])


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
    csv = input("Input the file path to the Yakkertech data: ")
    try:
        data = pd.read_csv(csv)
    except OSError as e:
        print("ERROR: File not found, try again")
        main()
    data = data.rename(
        columns={'RelSpeed': 'release_speed', 'HorzBreak': 'pfx_x', 'VertBreak': 'pfx_z', 'SpinAxis': 'spin_axis',
                 'SpinRate': 'release_spin_rate', 'PitcherThrows': 'p_throws', 'az0': 'az', 'ay0': 'ay', 'ax0': 'ax'})
    data['p_throws'] = data['p_throws'].replace({
        'Right': 0,
        'Left': 1
    })
    return data, csv


def load_data():
    sc_data = statcast('2023-04-01', '2023-05-31')

    data = sc_data[cols]
    data.dropna(subset=features, inplace=True)
    data['pitch_name'] = data['pitch_name'].replace({
        '4-Seam Fastball': 0,
        'Sinker': 1,
        'Cutter': 2,
        'Split-Finger': 3,
        'Slider': 4,
        'Curveball': 5,
        'Changeup': 6,
        'Sweeper': 4,
        'Forkball': 6,
        'Slurve': 4
    })

    data['p_throws'] = data['p_throws'].replace({
        'R': 0,
        'L': 1
    })

    drop_columns = ['Screwball', 'Other', 'Eephus', 'Slow Curve', 'Pitch Out', None, 'Knuckle Curve']
    data = data[~data['pitch_name'].isin(drop_columns)]

    X = data[features]
    y = data.pitch_name
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded


def convert_to_pname(pitch_type):
    pitch_name_mapping = {0: 'Fastball', 1: 'Sinker', 2: 'Cutter', 3: 'Splitter',
                          4: 'Slider', 5: 'Curveball', 6: 'Changeup'}

    if pitch_type is not None:
        prediction = pitch_type[0]

        pitch_name = pitch_name_mapping.get(prediction)

        if pitch_name not in pitch_types_yakker:
            pitch_name = None
    else:
        pitch_name = None

    return pitch_name


def train_model(X, y):
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
    model = XGBClassifier()
    model.fit(train_X, train_y)
    return model


def predict_rows(model, csv):
    predictions = []
    for index, row in csv.iterrows():
        if not row[features].isnull().any():
            pred = model.predict(row[features].values.reshape(1, -1))
        else:
            pred = None
        predictions.append(convert_to_pname(pred))
    csv["AutoPitchType"] = predictions
    return csv


def clean_rows(model, csv):
    for index, row in csv.iterrows():
        pitcher = row['Pitcher']
        if pitcher in pitcher_to_pitches:
            repertoire = pitcher_to_pitches[pitcher]
            pred_probs = model.predict_proba(row[features].values.reshape(1, -1))[0]
            valid_pitch_probs = [pred_probs[i] for i, pitch in enumerate(pitch_types_yakker) if pitch in repertoire]
            if valid_pitch_probs:
                max_prob_index = np.argmax(valid_pitch_probs)
                max_prob_pitch = [pitch for i, pitch in enumerate(pitch_types_yakker) if pitch in repertoire][
                    max_prob_index]
                csv.at[index, 'AutoPitchType'] = max_prob_pitch

    return csv


def adjust_changeups(csv):
    for index, row in csv.iterrows():
        if row['AutoPitchType'] == 'Changeup' and row['release_speed'] > 87 and row['TaggedPitchType'] == 'Fastball':
            csv.at[index, 'AutoPitchType'] = 'Fastball'
        if row['AutoPitchType'] == 'Changeup' and row['release_speed'] > 87 and row['TaggedPitchType'] == 'Sinker':
            csv.at[index, 'AutoPitchType'] = 'Sinker'
        if row['AutoPitchType'] == 'Slider' and row['release_speed'] > 87 and row['TaggedPitchType'] == 'Cutter':
            csv.at[index, 'AutoPitchType'] = 'Cutter'
        if row['AutoPitchType'] is None:
            csv.at[index, 'AutoPitchType'] = csv.at[index, 'TaggedPitchType']
    return csv


def main():
    ask_for_input()


if __name__ == "__main__":
    main()
