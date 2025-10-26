import pennylane as qml
import numpy as np

def amplitude_embed(x, wires):
    """Embedding por amplitude: aceita vetor 1D, faz pad se necessário."""
    x = np.asarray(x, dtype=float).ravel()
    qml.AmplitudeEmbedding(x, wires=wires, pad_with=0.0, normalize=True)
