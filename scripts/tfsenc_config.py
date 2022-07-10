import os


def create_output_directory(args):
    # output_prefix_add = '-'.join(args.emb_file.split('_')[:-1])

    # folder_name = folder_name + '-pca_' + str(args.reduce_to) + 'd'
    # full_output_dir = os.path.join(args.output_dir, folder_name)

    folder_name = "-".join([args.output_prefix, str(args.sid)])
    folder_name = folder_name.strip("-")
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
    
    PICKLE_DIR = os.path.join(
        os.getcwd(), "data", args.project_id, str(args.sid), "pickles"
    )
    path_dict = dict(PICKLE_DIR=PICKLE_DIR)

    stra = "cnxt_" + str(args.context_length)
    if args.emb_type == "glove50":
        stra = ""
        args.layer_idx = 1
    if args.emb_type == "blenderbot-small":
        stra = ""

    args.emb_file = "_".join(
        [
            str(args.sid),
            args.pkl_identifier,
            args.emb_type,
            stra,
            f"layer_{args.layer_idx:02d}",
            "embeddings.pkl",
        ]
    )
    args.load_emb_file = args.emb_file.replace("__", "_")

    args.signal_file = "_".join(
        [str(args.sid), args.pkl_identifier, "signal.pkl"]
    )
    args.electrode_file = "_".join([str(args.sid), "electrode_names.pkl"])
    args.stitch_file = "_".join(
        [str(args.sid), args.pkl_identifier, "stitch_index.pkl"]
    )

    args.output_dir = os.path.join(os.getcwd(), "results")
    args.full_output_dir = create_output_directory(args)

    args.best_lag = -1

    vars(args).update(path_dict)
    return args
