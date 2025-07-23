from signal_processing_for_participant import load_device_signals
from model_utils import train_model, plot_cl
import numpy as np

if __name__ == "__main__":
    participant_id = 3
    signals = load_device_signals(participant_id=participant_id, condition='baseline')
    bvp = signals['bvp']
    
    model, dataset = train_model(bvp, target=0.0, epochs=10)
    cl_vals = plot_cl(model, dataset, title=f"Baseline CL(t) — Participant {participant_id}")
    np.save(f"cl_baseline_p{participant_id}.npy", cl_vals)
