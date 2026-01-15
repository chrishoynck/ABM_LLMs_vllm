import ast
import math
import os
import pandas as pd
from datasets import load_dataset
ds = load_dataset("nvidia/Nemotron-Personas")
ds_small = ds["train"].shuffle(seed=42).select(range(10000))
df = ds_small.to_pandas()

# Clip every string field to max 120 chars
MAX_LEN = 200
NEW = True


def _clip_cell(x, n=MAX_LEN):
    if isinstance(x, str) and len(x) > n:
        if x[0] == '[' and x[-1] == ']':
            return x
        return x[:n]
    else:
        return x

def age_of_person(row):
    try:
        age = int(row['age'])
    except Exception:
        print(f"Could not convert age: {row['age']}")
        return False
    if age <= 16 or age >= 80:
        return False
    else:
        return True
    
df = df.map(_clip_cell)
df = df[df.apply(age_of_person, axis=1)]
columns_keep = ['persona', 'age', 'marital_status', 'hobbies_and_interests_list', 'skills_and_expertise_list','sex','bachelors_field', 'occupation', 'city' ]

if "personas_10k.csv" not in os.listdir("data/") or NEW:
    df.to_csv("data/personas_10k.csv", columns=columns_keep, index=False)
df = pd.read_csv("data/personas_10k.csv")


def parse_list_field(v):
    if pd.isna(v):
        return []
    v = str(v).strip()
    # JSON / Python-list style: ["a", "b", ...]
    if v.startswith("["):
        try:
            return [x.strip() for x in ast.literal_eval(v)]
        except Exception:
            pass
    # fallback: comma-separated
    return [x.strip() for x in v.split(",") if x.strip()]

def extract_name(persona_text):
    parts = persona_text.strip().split()
    return " ".join(parts[:2])  # first two words

def row_to_persona(row):
    return {
        "name": extract_name(row["persona"]),
        "persona_text": row["persona"],  # the long description text
        "age": int(row["age"]) if not math.isnan(row["age"]) else None,
        "sex": row["sex"],
        "marital_status": row["marital_status"],
        "bachelors_field": row["bachelors_field"],
        "occupation": row["occupation"],
        "city": row["city"],
        "hobbies": parse_list_field(row["hobbies_and_interests_list"]),
        "skills": parse_list_field(row["skills_and_expertise_list"]),
    }

def load_distorted_tweets(filepath="data/distorted_tweets.csv", numtweets=1000, seed=42):
    df = pd.read_csv(filepath)
    df_sampled = df.sample(n=numtweets, replace=True, random_state=seed)
    return df_sampled['tweet'].tolist()

def load_depressed_personas(filepath="data/depressed.csv", personass_to_load=1, seed=42):
    df = pd.read_csv(filepath)
    return [row_to_persona(row) for _, row in df.sample(n=personass_to_load, replace=True, random_state=seed).iterrows()]

def load_personas_from_file(filepath="data/personas_10k.csv", personass_to_load=10, seed=42):
    df = pd.read_csv(filepath)
    return [row_to_persona(row) for _, row in df.sample(n=personass_to_load, replace=False, random_state=seed).iterrows()]


def parse_phq9(row, dataset="H1"):
    return {
        "age": row[f'{dataset}_lft'],
        "phq9_sumscore": row[f'{dataset}_PHQ9_sumscore'],
        "depressive_symptoms": row[f'{dataset}_PHQ9_deprsymp'],
        "diagnosis": row[f'{dataset}_MDD'],
        "somber": row[f'{dataset}_WlbvSomber'],
        "joylessness": row[f'{dataset}_WlbvGeenPlezier'],
        "impaired_functioning": row[f'{dataset}_WlbvBelemmerd'],
        "Freq_depressive_episodes": row[f'{dataset}_WlbvFreqPeriode'],
        "Age_first_depressive_episode": row[f'{dataset}_WlbvLftdPeriode']
    }

def parse_phq9_cov(row):
    return {
    "interest_pleasure" : row["CovQ1_Depression_Enthusiasm"],
    "down_depressed": row["CovQ1_Depression_Dejection"],
    "insomnia": row["CovQ1_Depression_Insomnia"],
    "tired": row["CovQ1_Depression_Lethargy"],
    "appetite_loss": row["CovQ1_Depression_Appetite"],
    "failure_guilt": row["CovQ1_Depression_Failure"],
    "concentration_loss": row["CovQ1_Depression_Concentration"],
    "voice_low": row["CovQ1_Depression_Voice"],
    "nervousness": row["CovQ1_Depression_Nervousness"],
    "suicide": row["CovQ1_Depression_Suicide"]
    }

def load_phq9(filepath="data/confidential/phq9.sav", personass_to_load=10, seed=42):
    df = pd.read_spss(filepath)
    # print(df.columns)
    # print(df.columns[100:200])
    filtered = df.dropna(subset=['H1_PHQ9_sumscore', 'H1_PHQ9_deprsymp'])
    filtered.to_csv("data/confidential/phq9_filtered.csv", index=False)

    
    # print(df[ 'H2_PHQ9_sumscore', 'H2_PHQ9_deprsymp'])
    return [parse_phq9(row) for _, row in filtered.sample(n=personass_to_load, replace=False, random_state=seed).iterrows()]

# depressed_data = load_pghq9(personass_to_load=100)
# print(depressed_data[:5])

def write_phq9_to_file(filepath= "data/phq9/mood_data.csv", personas_to_write=1000):
    '''
    Write PHQ-9 data to CSV file
    Args:
        filepath (str): Path to the output CSV file
        personas_to_write (int): Number of personas to write
    '''
    data = load_phq9(personass_to_load=personas_to_write)
    panda_data = pd.DataFrame(data)
    panda_data.to_csv(filepath, index=False)


if __name__ == "__main__":
    write_phq9_to_file()
