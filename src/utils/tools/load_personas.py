import ast
import math
import os
import re
import pandas as pd
from datasets import load_dataset
from langdetect import detect


# Clip every string field to max 120 chars
MAX_LEN = 200
NEW = True

def parse_persona_traits():
    ds_pers = load_dataset("SynthLabsAI/PERSONA", split="train")
    ds_small = ds_pers.shuffle(seed=42).select(range(10000))
    df = ds_small.to_pandas()

    if "personas_traits_10k.csv" not in os.listdir("data/") or NEW:
        columns_keep = ['Age', 'Sex', 'Race', 'Occupation', 'Big Five Traits', 'Quirks', 'Personal Time', 'Lifestyle' ]
        df.to_csv("data/personas_traits_10k.csv", columns=columns_keep, index=False)
    df = pd.read_csv("data/personas_traits_10k.csv")
    
    print(df.columns)
    print(df.head())

def short_personas(seed=42):
    ds_pers = load_dataset("proj-persona/PersonaHub", 'persona', split="train")
    ds_small = ds_pers.shuffle(seed=seed).select(range(10000))
    df = ds_small.to_pandas()

    def clean_and_check_english(text):
        if not isinstance(text, str):
            return False
        try:
            is_en = detect(text) == 'en'
        except:
            is_en = False
        return is_en

    def clean_stubborn_quotes(text):
        clean = str(text)
        clean = clean.strip()
        while clean.startswith('"') or clean.startswith('“'):
            clean = clean[1:]
        while clean.endswith('"') or clean.endswith('”'):
            clean = clean[:-1]
        return clean
    
    df['persona'] = df['persona'].astype(str)
    df['persona'] = df['persona'].apply(clean_stubborn_quotes)
    df['persona'] = df['persona'].apply(lambda x: f'"{x}"')
    df = df[df['persona'].apply(clean_and_check_english)]

    if "personas_short_10k.csv" not in os.listdir("data/") or NEW:
        df.to_csv("data/personas_short_10k.csv", index=False)
        df = pd.read_csv("data/personas_short_10k.csv")

    print(df.columns)
    print(df.head())


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

def parse_persons(): 
    ds = load_dataset("nvidia/Nemotron-Personas")
    ds_small = ds["train"].shuffle(seed=42).select(range(10000))
    df = ds_small.to_pandas()
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

def load_happy_personas(filepath="data/happy_persona.csv", personass_to_load=1, seed=42):
    df = pd.read_csv(filepath)
    return [row_to_persona(row) for _, row in df.sample(n=personass_to_load, replace=True, random_state=seed).iterrows()]

# def load_personas_from_file(filepath="data/personas_short_10k.csv", personass_to_load=10, seed=42):
#     df = pd.read_csv(filepath)
#     return [row_to_persona(row) for _, row in df.sample(n=personass_to_load, replace=False, random_state=seed).iterrows()]

def load_personas_from_file(filepath="data/personas_short_10k.csv", personass_to_load=10, seed=42):
    df = pd.read_csv(filepath)
    return [row["persona"] for _, row in df.sample(n=personass_to_load, replace=False, random_state=seed).iterrows()]


# Default directories whose CSVs hold personas the eval pool must avoid.
# Qwen3.5-27B is the source of the existing PHQ-9 / BERT training data; expand
# the list if other model folders are also used as training data.
_DEFAULT_EVAL_EXCLUDE_DIRS = ["data/test_post/Qwen_Qwen3.5-27B"]


def load_or_build_persona_pool(
    n_needed: int,
    pool_path: str = "data/personas_eval_1000.csv",
    pool_size: int = 1000,
    source: str = "data/personas_short_10k.csv",
    exclude_dirs: list[str] | None = None,
    exclude_files: list[str] | None = None,
    seed: int = 1000,
):
    """Return the first `n_needed` personas from a shared eval pool, building the pool once.

    The pool is sampled a single time from `source`, excluding any persona that
    already appears in `exclude_dirs` (recursively scanned for *.csv files with
    a `persona` column) or `exclude_files`, then written to `pool_path`.
    Subsequent calls just read `pool_path` — so every model (local Qwen,
    Grok, anything else) sees the same fresh personas in the same order.

    Args:
        n_needed: how many personas the caller wants. Must be <= pool size.
        pool_path: cache file; if it exists it is used as-is (no re-sampling).
        pool_size: number of personas to sample on first build.
        source: source CSV with a `persona` column.
        exclude_dirs: dirs walked recursively for *.csv files whose personas
            should be excluded. Defaults to the Qwen3.5-27B training-data dir.
        exclude_files: extra explicit CSVs to exclude (same `persona` column).
        seed: RNG seed for the one-shot sample. Only affects which personas
            land in the pool on first build.

    Returns:
        List of persona strings, length `n_needed`, prefix of the pool.
    """
    if os.path.isfile(pool_path):
        df = pd.read_csv(pool_path)
        personas = df["persona"].astype(str).tolist()
        if n_needed > len(personas):
            raise ValueError(
                f"Pool at {pool_path} holds {len(personas)} personas but "
                f"{n_needed} were requested. Delete the file to rebuild at a "
                f"larger size, or lower `n_needed`."
            )
        print(f"[personas] using existing eval pool {pool_path} "
              f"({len(personas)} personas; taking first {n_needed})")
        return personas[:n_needed]

    exclude_paths: list[str] = list(exclude_files or [])
    for d in (exclude_dirs if exclude_dirs is not None else _DEFAULT_EVAL_EXCLUDE_DIRS):
        if not os.path.isdir(d):
            print(f"[personas] exclude dir not found, skipping: {d}")
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.endswith(".csv"):
                    exclude_paths.append(os.path.join(root, f))

    excluded: set[str] = set()
    for path in exclude_paths:
        try:
            df_ex = pd.read_csv(path, usecols=["persona"])
        except (ValueError, FileNotFoundError):
            # CSV without a `persona` column or missing — skip silently.
            continue
        excluded.update(df_ex["persona"].dropna().astype(str).tolist())

    df_all = pd.read_csv(source)
    df_pool = df_all[~df_all["persona"].astype(str).isin(excluded)]
    if len(df_pool) < pool_size:
        raise ValueError(
            f"Need {pool_size} fresh personas for the pool, only {len(df_pool)} "
            f"remain in {source} after excluding {len(excluded)} personas from "
            f"{len(exclude_paths)} file(s)."
        )
    if n_needed > pool_size:
        raise ValueError(f"n_needed={n_needed} exceeds pool_size={pool_size}.")

    sampled = df_pool.sample(n=pool_size, replace=False, random_state=seed)
    os.makedirs(os.path.dirname(os.path.abspath(pool_path)) or ".", exist_ok=True)
    sampled[["persona"]].to_csv(pool_path, index=False)
    personas = sampled["persona"].astype(str).tolist()
    print(f"[personas] built new eval pool: sampled {pool_size} from "
          f"{len(df_pool)}/{len(df_all)} (excluded {len(excluded)} personas "
          f"from {len(exclude_paths)} file(s)) → {pool_path}")
    return personas[:n_needed]

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
    short_personas()