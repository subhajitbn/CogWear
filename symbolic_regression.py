import numpy as np
from utils import extract_bvp_features
from model_utils import create_windows, normalize_windows
from pysr import PySRRegressor
from signal_processing_for_participant import load_device_signals

feature_names = ["mean", "std", "rms", "skew", "kurt", "peak_mean", "peak_std"]

def run_symbolic_regression(bvp_signals, cl_values, window_size=256, step_size=64):
    all_features = []

    for bvp_signal in bvp_signals:
        windows = create_windows(bvp_signal, window_size, step_size)
        windows = normalize_windows(windows)
        features = np.array([extract_bvp_features(w) for w in windows])
        all_features.append(features)

    X = np.concatenate(all_features, axis=0)
    y = cl_values[:len(X)]  # Just in case

    model = PySRRegressor(
        niterations=300,
        binary_operators=["+", "-", "*", "/"],
        # unary_operators=["square", "sqrt", "exp", "log", "abs"],
        unary_operators=["cos", "sin", "exp", "log", "abs"],
        model_selection="best",
        maxsize=30,
        verbosity=1,
    )

    model.fit(X, y, variable_names=feature_names)
    return model, X, y


if __name__ == "__main__":
    participant_id = 3

    print(f"\n[Symbolic Regression on JOINT BASELINE + COGNITIVE Data — Participant {participant_id}]\n")

    # Load BVP signals
    bvp_baseline = load_device_signals(participant_id, condition="baseline")["bvp"]
    bvp_cognitive = load_device_signals(participant_id, condition="cognitive_load")["bvp"]

    # Load corresponding CL(t)
    cl_baseline = np.load(f"cl_baseline_p{participant_id}.npy")  # should be ≈ 0s
    cl_cognitive = np.load(f"cl_cognitive_p{participant_id}.npy")  # should be ≈ 1s

    # Concatenate data
    bvp_all = [bvp_baseline, bvp_cognitive]
    cl_all = np.concatenate([cl_baseline, cl_cognitive])

    model, X, y_true = run_symbolic_regression(bvp_all, cl_all)

    # Show all discovered expressions
    print("\nAll Discovered Equations:")
    print(model)

    # Show best one
    print("\nBest symbolic expression:")
    print(model.get_best())

    import matplotlib.pyplot as plt

    # Predict using the symbolic model
    y_pred_pysr = model.predict(X)

    # Compute 1.5 - peak_mean
    peak_mean = X[:, feature_names.index("peak_mean")]
    y_pred_custom = 1.5 - peak_mean

    # Plot all together
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="True CL(t)", alpha=0.7)
    plt.plot(y_pred_pysr, label="Predicted CL(t) from symbolic model", alpha=0.7)
    plt.plot(y_pred_custom, label="1.5 - peak_mean", linestyle="--", alpha=0.7)
    plt.title(f"CL(t) Prediction — PySR vs Custom — Participant {participant_id}")
    plt.xlabel("Window Index")
    plt.ylabel("Cognitive Load (CL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
