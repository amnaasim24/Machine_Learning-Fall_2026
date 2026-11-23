from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'

class FailureMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

def load_models():
    required = ['preprocessor.pkl', 'random_forest.pkl', 'thresholds.pkl', 'neural_network.pt']
    missing = [name for name in required if not (MODELS / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing model files: ' + ', '.join(missing) + '\nRun the notebook through its final save-model cell first.')
    with open(MODELS / 'preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open(MODELS / 'random_forest.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open(MODELS / 'thresholds.pkl', 'rb') as f:
        thresholds = pickle.load(f)
    checkpoint = torch.load(MODELS / 'neural_network.pt', map_location='cpu', weights_only=True)
    model = FailureMLP(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return preprocessor, rf, thresholds, model

def ask_number(prompt, low, high):
    while True:
        try:
            value = float(input(f'{prompt} ({low}–{high}): ').strip())
            if np.isfinite(value) and low <= value <= high:
                return value
        except ValueError:
            pass
        print(f'Please enter a number between {low} and {high}.')

def main():
    preprocessor, rf, thresholds, model = load_models()
    print('\nMACHINE FAILURE PREDICTION — educational demo')
    while True:
        machine_type = input('Machine type (L, M, H): ').strip().upper()
        if machine_type in ('L', 'M', 'H'):
            break
        print('Please enter L, M, or H.')
    values = {
        'Type': machine_type,
        'Air temperature [K]': ask_number('Air temperature [K]', 250, 350),
        'Process temperature [K]': ask_number('Process temperature [K]', 250, 370),
        'Rotational speed [rpm]': ask_number('Rotational speed [rpm]', 0, 5000),
        'Torque [Nm]': ask_number('Torque [Nm]', 0, 150),
        'Tool wear [min]': ask_number('Tool wear [min]', 0, 300),
    }
    features = pd.DataFrame([values])
    encoded = preprocessor.transform(features).astype(np.float32)
    rf_prob = float(rf.predict_proba(encoded)[0, 1])
    with torch.no_grad():
        nn_prob = float(torch.sigmoid(model(torch.from_numpy(encoded))).item())
    print('\nPREDICTION RESULTS')
    for name, probability, threshold in [
        ('Random Forest', rf_prob, thresholds['random_forest']),
        ('PyTorch ANN', nn_prob, thresholds['neural_network']),
    ]:
        status = 'Potential failure' if probability >= threshold else 'Normal operation'
        print(f'{name}: {status} | Score: {probability:.2%} | Threshold: {threshold:.2f}')
    print('\nNote: Trained on synthetic data; not suitable for real-world safety decisions.')

if __name__ == '__main__':
    main()
