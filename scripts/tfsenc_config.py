import os


def create_output_directory(args):
    # output_prefix_add = '-'.join(args.emb_file.split('_')[:-1])

    # folder_name = folder_name + '-pca_' + str(args.reduce_to) + 'd'
    # full_output_dir = os.path.join(args.output_dir, folder_name)

    folder_name = "-".join([args.output_prefix, str(args.sid)]).strip("-")

    if args.model_mod:
        parent_folder_name = "-".join([args.output_parent_dir, args.model_mod])
    else:
        parent_folder_name = args.output_parent_dir
    full_output_dir = os.path.join(
        os.getcwd(), "results", args.project_id, parent_folder_name, folder_name
    )

    os.makedirs(full_output_dir, exist_ok=True)

    return full_output_dir


def setup_environ(args):
    """Update args with project specific directories and other flags"""

    args.emb_type = args.emb_type.split("/")[-1]
    args.align_with = [model.split("/")[-1] for model in args.align_with]

    if "glove50" in args.align_with:
        args.align_with[args.align_with.index("glove50")] = "glove"

    INPUT_DIR = os.path.join(os.getcwd(), "data", args.project_id, str(args.sid))

    args.PICKLE_DIR = os.path.join(INPUT_DIR, "pickles")
    EMB_DIR = os.path.join(args.PICKLE_DIR, "embeddings")
    MODEL_EMB_DIR = os.path.join(EMB_DIR, args.emb_type, args.pkl_identifier)

    if args.emb_type == "glove50":
        args.emb_type = "glove"
        args.layer_idx = 1
        args.context_length = 1

    args.base_df_path = os.path.join(MODEL_EMB_DIR, "base_df.pkl")

    args.emb_df_path = os.path.join(
        MODEL_EMB_DIR,
        f"cnxt_{args.context_length:04d}",
        f"layer_{args.layer_idx:02d}.pkl",
    )

    args.signal_file = "_".join([str(args.sid), args.pkl_identifier, "signal.pkl"])
    args.electrode_file = "_".join([str(args.sid), "electrode_names.pkl"])
    args.stitch_file = "_".join(
        [str(args.sid), args.pkl_identifier, "stitch_index.pkl"]
    )

    args.output_dir = os.path.join(os.getcwd(), "results")
    args.full_output_dir = create_output_directory(args)

    args.best_lag = -1

    return args
