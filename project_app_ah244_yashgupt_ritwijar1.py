import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_add_pool
from rdkit import Chem, rdBase
from rdkit.Chem import Draw, AllChem
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="GRASP: Molecular Property Predictor",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3, h4, h5, h6 { color: #004AAD; }
    .st-emotion-cache-10trblm { color: #004AAD; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        border: 2px solid #FF4B2B; background-color: #FF4B2B; color: white;
        font-weight: bold; border-radius: 10px; transition: all 0.3s; padding: 10px 20px;
    }
    .stButton>button:hover { background-color: #E84123; border-color: #E84123; }
    .result-card {
        background-color: white; padding: 25px; border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 20px;
        border-top: 5px solid #FF4B2B; text-align: center;
    }
    .result-title { font-size: 1.3em; font-weight: bold; color: #004AAD; margin-bottom: 10px; }
    .result-value { font-size: 2.2em; font-weight: 700; color: #333333; }
    .result-interpretation { font-size: 1em; color: #555555; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

rdBase.DisableLog('rdApp.*')

GRAPH_EMB_DIM = 128
GRAPH_LAYERS = 4
ATOM_FEATURE_MAP = {
    'atomic_num': list(range(1, 119)), 'degree': list(range(6)), 
    'formal_charge': list(range(-2, 3)), 'hybridization': [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ], 'is_aromatic': [0, 1], 'is_in_ring': [0, 1]
}

def get_atom_features(atom):
    features = []
    def one_hot_encode(value, choices):
        encoding = [0] * (len(choices) + 1)
        encoding[choices.index(value) if value in choices else -1] = 1
        return encoding
    features.extend(one_hot_encode(atom.GetAtomicNum(), ATOM_FEATURE_MAP['atomic_num']))
    features.extend(one_hot_encode(atom.GetDegree(), ATOM_FEATURE_MAP['degree']))
    features.extend(one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURE_MAP['formal_charge']))
    features.extend(one_hot_encode(atom.GetHybridization(), ATOM_FEATURE_MAP['hybridization']))
    features.extend(one_hot_encode(int(atom.GetIsAromatic()), ATOM_FEATURE_MAP['is_aromatic']))
    features.extend(one_hot_encode(int(atom.IsInRing()), ATOM_FEATURE_MAP['is_in_ring']))
    return torch.tensor(features, dtype=torch.float)

def smiles_to_graph_data(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return None
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_features: return None
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
        graph_data = Data(x=torch.stack(atom_features), 
                          edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous())
        if graph_data.edge_index.dim() == 1:
            graph_data.edge_index = graph_data.edge_index.view(2, -1)
        return graph_data
    except Exception:
        return None

def smiles_to_image(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None
    AllChem.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=(400, 300))

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, embedding_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers
        num_node_features = sum(len(c) + 1 for c in ATOM_FEATURE_MAP.values())
        for i in range(num_layers):
            in_dim = num_node_features if i == 0 else embedding_dim
            mlp = nn.Sequential(nn.Linear(in_dim, 2 * embedding_dim), nn.ReLU(), nn.Linear(2 * embedding_dim, embedding_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(embedding_dim))
    def forward(self, x, edge_index, batch):
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return global_add_pool(h, batch)

class DownstreamModel(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        num_node_features = sum(len(c) + 1 for c in ATOM_FEATURE_MAP.values())
        self.graph_encoder = GraphEncoder(num_node_features, GRAPH_EMB_DIM, GRAPH_LAYERS, 0.1)
        self.prediction_head = nn.Sequential(nn.Linear(GRAPH_EMB_DIM, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_tasks))
    def forward(self, data_batch):
        return self.prediction_head(self.graph_encoder(data_batch.x, data_batch.edge_index, data_batch.batch))

@st.cache_resource
def load_all_models():
    device = torch.device('cpu')
    models = {}
    model_configs = {
        'bbbp': {'tasks': 1, 'file': 'best_bbbp.pt'},
        'esol': {'tasks': 1, 'file': 'best_esol.pt'},
        'tox21': {'tasks': 12, 'file': 'best_tox21.pt'}
    }
    for name, config in model_configs.items():
        model = DownstreamModel(config['tasks']).to(device)
        model.load_state_dict(torch.load(config['file'], map_location=device))
        model.eval()
        models[name] = model
    return models

def get_predictions(smiles_string, models):
    graph_data = smiles_to_graph_data(smiles_string)
    if graph_data is None: return None
    graph_batch = Batch.from_data_list([graph_data])
    
    with torch.no_grad():
        pred_bbbp = torch.sigmoid(models['bbbp'](graph_batch)).item()
        pred_esol = models['esol'](graph_batch).item()
        pred_tox21 = torch.sigmoid(models['tox21'](graph_batch)).numpy().flatten()
    
    return {'bbbp': pred_bbbp, 'esol': pred_esol, 'tox21': pred_tox21}

def interpret_results(predictions):
    bbbp_score = predictions['bbbp']
    esol_score = predictions['esol']
    tox_avg = np.mean(predictions['tox21'])
    
    bbbp_interp = "✅ Likely to Cross" if bbbp_score > 0.5 else "❌ Unlikely to Cross"
    esol_interp_main = "High" if esol_score > -2 else ("Moderate" if esol_score > -4 else "Low")
    tox_interp = "☠️ High Risk" if tox_avg > 0.4 else ("⚠️ Medium Risk" if tox_avg > 0.2 else "✅ Low Risk")
    
    return bbbp_interp, esol_interp_main, tox_interp

st.title("🔬 GRASP Molecular Property Predictor")
st.markdown("An application of self-supervised learning for predicting key chemical properties from a SMILES string.")

try:
    models = load_all_models()
except FileNotFoundError:
    st.error("Model files (best_bbbp.pt, best_esol.pt, best_tox21.pt) not found. Please place them in the app directory.")
    st.stop()

with st.sidebar:
    st.header("About GRASP")
    st.info("This app uses a Graph Neural Network pre-trained on 500,000 unlabeled molecules. This knowledge transfer allows it to make accurate predictions on new molecules.")
    st.header("Example SMILES")
    st.code("CCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O", language="smiles")

smiles_input = st.text_input("Enter SMILES String:", "CC(=O)Oc1ccccc1C(=O)O", help="e.g., CC(=O)Oc1ccccc1C(=O)O for Aspirin")

if st.button("Predict Properties"):
    if not smiles_input:
        st.warning("Please enter a SMILES string.")
    else:
        with st.spinner('Analyzing molecule...'):
            image = smiles_to_image(smiles_input)
            predictions = get_predictions(smiles_input, models) if image else None

        if predictions:
            st.success("Prediction Complete!")
            
            interpretations = interpret_results(predictions)
            bbbp_interp, esol_interp_main, tox_interp = interpretations

            col_img, col_results = st.columns([2, 3])
            
            with col_img:
                st.subheader("Molecule Structure")
                st.image(image, use_container_width=True)

            with col_results:
                st.subheader("Predicted Properties")
                
                esol_value_str = f"(logS = {predictions['esol']:.2f})"
                
                st.markdown(f'<div class="result-card"><div class="result-title">Blood-Brain Barrier</div><p class="result-value">{predictions["bbbp"]:.1%}</p><p class="result-interpretation">{bbbp_interp}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-card"><div class="result-title">💧 Water Solubility</div><p class="result-value">{esol_interp_main}</p><p class="result-interpretation">{esol_value_str}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-card"><div class="result-title">Overall Toxicity Risk</div><p class="result-value">{np.mean(predictions["tox21"]):.1%}</p><p class="result-interpretation">{tox_interp}</p></div>', unsafe_allow_html=True)

            with st.expander("Show Detailed Breakdown by Toxicity Pathway"):
                tox21_tasks = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
                tox_df = pd.DataFrame({'Toxicity Pathway': tox21_tasks, 'Predicted Probability': predictions["tox21"]})
                st.bar_chart(tox_df.set_index('Toxicity Pathway'))
        else:
            st.error("Invalid or unsupported SMILES string. Please check the format and try again.")