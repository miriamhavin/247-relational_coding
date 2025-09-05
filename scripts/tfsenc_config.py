import argparse
import getpass
import os
import argparse, yaml, getpass
from pathlib import Path
import numpy as np
import torch
import yaml
from himalaya.backend import set_backend
from utils import get_git_hash
from ast import literal_eval

ELEC_SIGNAL_PREPROCESS_MAP = {
    "podcast": dict.fromkeys(
        [
            "661",
            "662",
            "717",
            "723",
            "737",
            "741",
            "742",
            "743",
            "763",
            "798",
            "777",
        ],
        "preprocessed_all",
    ),
    "tfs": dict.fromkeys(["625", "676"], "preprocessed")
    | dict.fromkeys(["7170"], "preprocessed_v2")
    | dict.fromkeys(["798"], "preprocessed_allElec"),
}

ELEC_SIGNAL_FOLDER_MAP = {
    "podcast": "/projects/HASSON/247/data/podcast-data",
    "tfs": "/projects/HASSON/247/data/conversations-car",
}


def clean_lm_model_name(item):
    """Remove unnecessary parts from the language model name.

    Args:
        item (str/list): full model name from HF Hub

    Returns:
        (str/list): pretty model name

    Example:
        clean_lm_model_name(EleutherAI/gpt-neo-1.3B) == 'gpt-neo-1.3B'
    """
    if isinstance(item, str):
        return item.split("/")[-1]

    if isinstance(item, list):
        return [clean_lm_model_name(i) for i in item]

    print("Invalid input. Please check.")



def parse_arguments():
    """Read arguments from yaml config file (+ optional CLI overrides)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", nargs="*", type=str,
                        default=["configs/config.yml"])
    # optional CLI overrides (None means "use YAML/default")
    parser.add_argument("--min-occ", dest="min_occ", type=int, default=None)
    parser.add_argument("--B-perm-cols", dest="B_perm_cols", type=int, default=None)
    parser.add_argument("--B-mantel", dest="B_mantel", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lags", type=str, default=None,
                        help="Python-style list of lags, e.g. '[-500,0,500]'")
    parser.add_argument("--output-dir-name", dest="output_dir_name", type=str, default=None,
                        help="Folder name suffix for this run, used in results path")

    args_cli = parser.parse_args()

    # ---- merge YAMLs (later files override earlier ones) ----
    all_yml_args = {}
    for config_file in args_cli.config_file:
        with open(config_file, "r") as f:
            yml = yaml.safe_load(f) or {}
            all_yml_args |= yml

    # ---- hard defaults (used if not in YAML and no CLI override) ----
    defaults = {
        "min_occ": 50,          # match your earlier successful run
        "B_perm_cols": 2000,    # per-electrode perms
        "B_mantel": 5000,       # Mantel perms per electrode
        "seed": 42,
        "output_dir_name": "default",
    }
    for k, v in defaults.items():
        all_yml_args.setdefault(k, v)

    # ---- CLI overrides take precedence over YAML ----
    for k in ["min_occ", "B_perm_cols", "B_mantel", "seed", "lags", "output_dir_name"]:
        v = getattr(args_cli, k, None)
        if v is not None:
            all_yml_args[k] = v

    # ---- add metadata ----
    all_yml_args["user_id"] = getpass.getuser()
    all_yml_args["git_hash"] = get_git_hash()

    # ---- normalize / type-coerce a few fields ----
    if "conv_ids" in all_yml_args:
        v = all_yml_args["conv_ids"]
        if isinstance(v, str):
            try:
                v = eval(v, {"np": __import__("numpy")})   # safely allow np.arange
            except Exception:
                pass
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, (int, np.integer)):
            v = [int(v)]
        all_yml_args["conv_ids"] = v
        
    for key in ("elecs", "lags"):
        if key in all_yml_args and isinstance(all_yml_args[key], str):
            try:
                all_yml_args[key] = eval(all_yml_args[key])
            except Exception:
                pass

    # model name cleanup
    if all_yml_args.get("emb") == "glove50":
        all_yml_args["layer_idx"] = 1
        all_yml_args["context_length"] = 1
    else:
        if "emb" in all_yml_args:
            all_yml_args["emb"] = clean_lm_model_name(all_yml_args["emb"])

    # ensure output_dir exists if present
    outdir = all_yml_args.get("output_dir")
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    # return as Namespace + the raw dict
    return argparse.Namespace(**all_yml_args), all_yml_args


def setup_environ(args):
    """
    Update args with project specific directories and other flags

    Args:
        args (namespace): arguments

    Returns:
        args (namespace): arguments plus directory paths
    """

    # input directory paths (pickles)
    DATA_DIR = os.path.join(os.getcwd(), "data")
    PICKLE_DIR = os.path.join(DATA_DIR, args.project_id, str(args.sid), "pickles")
    EMB_DIR = os.path.join(PICKLE_DIR, "embeddings", args.emb, "full")
    args.base_df_path = os.path.join(EMB_DIR, "base_df.pkl")
    args.emb_df_path = os.path.join(
        EMB_DIR,
        f"cnxt_{args.context_length:04d}",
        f"layer_{args.layer_idx:02d}.pkl",
    )
    args.electrode_file_path = os.path.join(
        PICKLE_DIR, ("_".join([str(args.sid), "electrode_names.pkl"]))
    )
    if getattr(args, "sig_elec_file_prod", None):
        args.sig_elec_file_prod = os.path.join(os.getcwd(), args.sig_elec_file_prod)
    if getattr(args, "sig_elec_file_comp", None):
        args.sig_elec_file_comp = os.path.join(os.getcwd(), args.sig_elec_file_comp)
    args.stitch_file_path = os.path.join(
        PICKLE_DIR, ("_".join([str(args.sid), "full_stitch_index.pkl"]))
    )

    # input directory paths (elec mat files)
    ELEC_SIGNAL_DIR = ELEC_SIGNAL_FOLDER_MAP[args.project_id]
    args.elec_signal_file_path = os.path.join(
        ELEC_SIGNAL_DIR, str(args.sid), "NY*Part*conversation*"
    )
    args.elec_signal_process_flag = ELEC_SIGNAL_PREPROCESS_MAP[args.project_id][
        str(args.sid)
    ]

    # output directory paths
    OUTPUT_DIR = os.path.join(os.getcwd(), "results", args.project_id)
    RESULT_PARENT_DIR = f"{args.user_id[0:2]}-{args.project_id}-{args.sid}-{args.emb}-{args.output_dir_name}"
    RESULT_CHILD_DIR = f"{args.user_id[0:2]}-{args.window_size}ms-{args.sid}"
    args.output_dir = os.path.join(OUTPUT_DIR, RESULT_PARENT_DIR, RESULT_CHILD_DIR)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        print("set backend to cuda")
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        print("set backend to cpu numpy")
        backend = set_backend("numpy", on_error="warn")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # HACK

    return args


def write_config(args, yml_args):
    """Write configuration to a file

    Args:
        args (namespace): configuration
        yml_args (dict): original yml args
    """
    config_file_path = os.path.join(args.output_dir, "config.yml")
    with open(config_file_path, "w") as outfile:
        yaml.dump(yml_args, outfile, default_flow_style=False, sort_keys=False)
