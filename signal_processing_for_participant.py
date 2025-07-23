import os
import numpy as np
import pandas as pd
from scipy.signal import resample

def load_device_signals(participant_id, data_root="pilot", device="empatica", physioparam="bvp", condition="baseline"):
    """
    Loads BVP signals for a single participant and condition.
    
    Args:
        participant_id (str or int): Participant folder name
        data_root (str): Top-level folder (e.g., 'pilot')
        condition (str): 'baseline' or 'cognitive_load'
        
    Returns:
        dict: {'bvp': ..., 'eda': ..., 'temp': ...}, each a 1D np.array aligned at 4 Hz
    """
    
    devices = ["empatica", "samsung", "muse"]

    pid = str(participant_id)
    base_path = os.path.join(data_root, pid, condition)

    
    ebvp = pd.read_csv(os.path.join(base_path, "empatica_bvp.csv"))["bvp"].values
    # Trim 2 sec at start & end from BVP to match EDA/TEMP preprocessing
    sec_remove = 2
    ebvp = ebvp[64 * sec_remove : -64 * sec_remove]  # 64 Hz

    return {
        "bvp": ebvp
    }



def save_empatica_signals_to_csv(signals, participant_id, condition, output_dir="processed_csv"):
    """
    Saves aligned BVP, EDA, TEMP signals into a single CSV for a given participant and condition.

    Args:
        signals (dict): {'bvp': np.array, 'eda': np.array, 'temp': np.array}
        participant_id (str or int): Participant ID
        condition (str): 'baseline' or 'cognitive_load'
        output_dir (str): Output folder path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame({
        "bvp": signals["bvp"],
        "eda": signals["eda"],
        "temp": signals["temp"]
    })

    filename = f"participant_{participant_id}_{condition}.csv"
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    
    print(f"✅ Saved: {path}")
