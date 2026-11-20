import os
import pickle
import time
from collections import OrderedDict
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

DATASET_NAME = "meta"   # "drugbank" 或 "meta"

BASE_DIR = r"D:\software\python\python-learn\GSMeta-DDIE"
CODE_DIR = os.path.join(BASE_DIR, "GSMeta-DDIE")
DATA_DIR = os.path.join(BASE_DIR, "METADDIEdata")

DATASET_PATHS = {
    "drugbank": {
        "input_csv": os.path.join(DATA_DIR, "sqlite_exports", "drugbank_drug", "drug_smiles.csv"),
        "output_pkl": os.path.join(DATA_DIR, "drugbank_drug_struct_meta.pkl"),
    },
    "meta": {
        "input_csv": os.path.join(DATA_DIR, "sqlite_exports", "Drug_META_DDIE", "drug.csv"),
        "output_pkl": os.path.join(DATA_DIR, "meta_ddie_drug_struct_meta.pkl"),
    },
}

def detect_id_col(df: pd.DataFrame) -> str:
    candidates = ["Drug_ID", "drug_id", "id", "drugbank_id", "drugid"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"找不到药物ID列，当前列名: {df.columns.tolist()}")


def detect_smiles_col(df: pd.DataFrame) -> str:
    candidates = ["SMILES", "smiles", "smile", "canonical_smiles"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"找不到 SMILES 列，当前列名: {df.columns.tolist()}")

def safe_mol_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def get_scaffold_key(mol):
    """
    用 Murcko scaffold 作为 scaffold key
    """
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaf is None or scaf == "":
            scaf = Chem.MolToSmiles(mol, canonical=True)
        return scaf
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True)


def get_brics_signature(mol, max_frags=12, max_atoms_for_brics=120):
    """
    motif 用 BRICS 片段签名。
    对特别大的分子直接降级，避免 BRICS 过慢。
    """
    try:
        if mol.GetNumAtoms() > max_atoms_for_brics:
            return f"LARGE_MOL_ATOMS_{mol.GetNumAtoms()}"

        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        frags = []

    if len(frags) == 0:
        try:
            frags = [Chem.MolToSmiles(mol, canonical=True)]
        except Exception:
            frags = ["INVALID"]

    frags = sorted(frags)[:max_frags]
    return "||".join(frags)

def _bucketize(v, bins):
    """
    把连续值离散成小桶，避免 role_key 过细碎
    """
    for i, b in enumerate(bins):
        if v <= b:
            return i
    return len(bins)

def get_role_key(mol):
    """
    role_key: 用分子整体拓扑/理化形态定义“角色”
    与 scaffold / motif 独立
    对超大分子做降级，避免计算过慢。
    """
    num_atoms = mol.GetNumAtoms()

    if num_atoms > 150:
        return f"LARGE_ROLE|A{num_atoms}"

    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    num_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    num_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Descriptors.MolLogP(mol)

    role_key = (
        f"A{_bucketize(num_atoms, [10, 20, 30, 40, 60, 100])}"
        f"|R{_bucketize(num_rings, [0, 1, 2, 3, 4, 6])}"
        f"|AR{_bucketize(num_arom_rings, [0, 1, 2, 3, 4])}"
        f"|H{_bucketize(num_hetero, [1, 3, 5, 8, 12])}"
        f"|RB{_bucketize(num_rot, [0, 2, 4, 8, 12])}"
        f"|HBA{_bucketize(hba, [0, 2, 5, 8, 12])}"
        f"|HBD{_bucketize(hbd, [0, 1, 2, 3, 5])}"
        f"|TPSA{_bucketize(tpsa, [20, 60, 100, 140, 200])}"
        f"|LOGP{_bucketize(logp, [0, 1, 3, 5, 7])}"
    )
    return role_key

def build_struct_meta(input_csv: str, output_pkl: str):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")

    df = pd.read_csv(input_csv)

    id_col = detect_id_col(df)
    smiles_col = detect_smiles_col(df)

    print(f"检测到药物ID列: {id_col}")
    print(f"检测到 SMILES 列: {smiles_col}")
    print(f"总行数: {len(df)}")

    scaffold_vocab = OrderedDict()
    motif_vocab = OrderedDict()
    role_vocab = OrderedDict()
    drug_struct_meta = {}

    n_valid = 0
    n_invalid = 0

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        t0 = time.time()

        drug_id = str(getattr(row, id_col)).strip()
        smiles = str(getattr(row, smiles_col)).strip()

        print(f"[{idx}/{len(df)}] start {drug_id}", flush=True)

        # 跳过空字符串 / nan
        if smiles.lower() in ["nan", "none", ""]:
            drug_struct_meta[drug_id] = {
                "valid": False,
                "canonical_smiles": None,
                "scaffold_key": None,
                "scaffold_id": -1,
                "motif_key": None,
                "motif_cluster_id": -1,
                "role_key": None,
                "role_id": -1,
            }
            n_invalid += 1

        else:
            mol = safe_mol_from_smiles(smiles)

            if mol is None:
                drug_struct_meta[drug_id] = {
                    "valid": False,
                    "canonical_smiles": None,
                    "scaffold_key": None,
                    "scaffold_id": -1,
                    "motif_key": None,
                    "motif_cluster_id": -1,
                    "role_key": None,
                    "role_id": -1,
                }
                n_invalid += 1
            else:
                can_smiles = Chem.MolToSmiles(mol, canonical=True)

                scaffold_key = get_scaffold_key(mol)
                motif_key = get_brics_signature(mol)  # 这里已经带大分子降级
                role_key = get_role_key(mol)  # 这里也已经带大分子降级

                if scaffold_key not in scaffold_vocab:
                    scaffold_vocab[scaffold_key] = len(scaffold_vocab)

                if motif_key not in motif_vocab:
                    motif_vocab[motif_key] = len(motif_vocab)

                if role_key not in role_vocab:
                    role_vocab[role_key] = len(role_vocab)

                scaffold_id = scaffold_vocab[scaffold_key]
                motif_cluster_id = motif_vocab[motif_key]
                role_id = role_vocab[role_key]

                drug_struct_meta[drug_id] = {
                    "valid": True,
                    "canonical_smiles": can_smiles,
                    "scaffold_key": scaffold_key,
                    "scaffold_id": scaffold_id,
                    "motif_key": motif_key,
                    "motif_cluster_id": motif_cluster_id,
                    "role_key": role_key,
                    "role_id": role_id,
                }
                n_valid += 1

        dt = time.time() - t0
        if dt > 5:
            print(f"WARNING: slow molecule {drug_id}, time={dt:.2f}s", flush=True)

        print(f"[{idx}/{len(df)}] done  {drug_id}  time={dt:.2f}s", flush=True)

        if idx % 10 == 0 or idx == len(df):
            print(f"已处理 {idx}/{len(df)} | valid={n_valid} | invalid={n_invalid}", flush=True)

    payload = {
        "drug_struct_meta": drug_struct_meta,
        "scaffold_vocab": dict(scaffold_vocab),
        "motif_vocab": dict(motif_vocab),
        "role_vocab": dict(role_vocab),
        "meta_info": {
            "dataset_name": DATASET_NAME,
            "input_csv": input_csv,
            "num_drugs": len(drug_struct_meta),
            "num_valid": n_valid,
            "num_invalid": n_invalid,
            "num_scaffolds": len(scaffold_vocab),
            "num_motif_clusters": len(motif_vocab),
            "num_roles": len(role_vocab),
        }
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n保存完成: {output_pkl}")
    print(payload["meta_info"])

    print("\n样例前5个：")
    sample_keys = list(drug_struct_meta.keys())[:5]
    for k in sample_keys:
        print(k, drug_struct_meta[k])

    print("\n===== build_drug_struct_meta.py 已全部执行完成 =====")


if __name__ == "__main__":
    if DATASET_NAME not in DATASET_PATHS:
        raise ValueError(f"未知数据集: {DATASET_NAME}, 可选: {list(DATASET_PATHS.keys())}")

    input_csv = DATASET_PATHS[DATASET_NAME]["input_csv"]
    output_pkl = DATASET_PATHS[DATASET_NAME]["output_pkl"]

    print(f"当前数据集: {DATASET_NAME}")
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_pkl}")

    build_struct_meta(input_csv, output_pkl)