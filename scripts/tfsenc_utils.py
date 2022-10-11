import csv
import os
from functools import partial
from imp import C_EXTENSION
from multiprocessing import Pool

import mat73
import numpy as np
from numba import jit, prange
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline


def encColCorr(CA, CB):
    """[summary]

    Args:
        CA ([type]): [description]
        CB ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = np.shape(CA)[0] - 2

    CA -= np.mean(CA, axis=0)
    CB -= np.mean(CB, axis=0)

    r = np.sum(CA * CB, 0) / np.sqrt(np.sum(CA * CA, 0) * np.sum(CB * CB, 0))

    t = r / np.sqrt((1 - np.square(r)) / df)
    p = stats.t.sf(t, df)

    r = r.squeeze()

    if r.size > 1:
        r = r.tolist()
    else:
        r = float(r)

    return r, p, t


# def cv_lm_003(X, Y, folds, fold_num, lag):
#     """Cross-validated predictions from a regression model using sequential
#         block partitions with nuisance regressors included in the training
#         folds

#     Args:
#         X ([type]): [description]
#         Y ([type]): [description]
#         kfolds ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     if lag == -1:
#         print("running normal encoding")
#     else:
#         print("running best-lag")

#     # Data size
#     nSamps = X.shape[0]
#     nChans = Y.shape[1] if Y.shape[1:] else 1

#     YHAT = np.zeros((nSamps, nChans))

#     # Go through each fold, and split
#     for i in range(0, fold_num):
#         # Shift the number of folds for this iteration
#         # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
#         #                       ^ dev fold
#         #                         ^ test fold
#         #                 | - | <- train folds

#         # Extract each set out of the big matricies
#         Xtra, Xtes = X[folds == i], X[folds != i]
#         Ytra, Ytes = Y[folds == i], Y[folds != i]

#         # Mean-center
#         Xtra -= np.mean(Xtra, axis=0)
#         Xtes -= np.mean(Xtes, axis=0)
#         Ytra -= np.mean(Ytra, axis=0)
#         Ytes -= np.mean(Ytes, axis=0)

#         # Fit model
#         model = make_pipeline(PCA(50, whiten=True), LinearRegression())
#         model.fit(Xtra, Ytra)

#         if lag != -1:  # best-lag model
#             B = model.named_steps["linearregression"].coef_
#             assert lag < B.shape[0], f"Lag index out of range"
#             B = np.repeat(B[lag, :][np.newaxis, :], B.shape[0], 0)
#             model.named_steps["linearregression"].coef_ = B

#         # Predict
#         foldYhat = model.predict(Xtes)

#         # Add to data matrices
#         YHAT[folds != i, :] = foldYhat.reshape(-1, nChans)

#     return YHAT


def cv_lm_003_prod_comp(args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, lag):
    if lag == -1:
        print("running regression")
    else:
        print("running regression with best_lag")

    nSamps = Xtes.shape[0]
    nChans = Ytra.shape[1] if Ytra.shape[1:] else 1

    YHAT = np.zeros((nSamps, nChans))
    Ynew = np.zeros((nSamps, nChans))

    for i in range(0, args.fold_num):
        Xtraf, Xtesf = Xtra[fold_tra != i], Xtes[fold_tes == i]
        Ytraf, Ytesf = Ytra[fold_tra != i], Ytes[fold_tes == i]

        Xtraf -= np.mean(Xtra, axis=0)
        Xtesf -= np.mean(Xtes, axis=0)
        Ytraf -= np.mean(Ytra, axis=0)
        Ytesf -= np.mean(Ytes, axis=0)

        # Fit model
        model = make_pipeline(PCA(50, whiten=True), LinearRegression())
        model.fit(Xtraf, Ytraf)

        if lag != -1:
            B = model.named_steps["linearregression"].coef_
            assert lag < B.shape[0], f"Lag index out of range"
            B = np.repeat(B[lag, :][np.newaxis, :], B.shape[0], 0)  # best-lag model
            model.named_steps["linearregression"].coef_ = B

        # Predict
        foldYhat = model.predict(Xtesf)

        Ynew[fold_tes == i, :] = Ytesf.reshape(-1, nChans)
        YHAT[fold_tes != i, :] = foldYhat.reshape(-1, nChans)

    return YHAT


# @jit(nopython=True)
# def fit_model(X, y):
#     """Calculate weight vector using normal form of regression.

#     Returns:
#         [type]: (X'X)^-1 * (X'y)
#     """
#     beta = np.linalg.solve(X.T.dot(X), X.T.dot(y))
#     return beta


@jit(nopython=True)
def build_Y(onsets, convo_onsets, convo_offsets, brain_signal, lags, window_size):
    """[summary]

    Args:
        onsets ([type]): [description]
        brain_signal ([type]): [description]
        lags ([type]): [description]
        window_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    half_window = round((window_size / 1000) * 512 / 2)

    # Y1 = np.zeros((len(onsets), len(lags), 2 * half_window + 1))
    Y1 = np.zeros((len(onsets), len(lags)))

    for lag in prange(len(lags)):

        lag_amount = int(lags[lag] / 1000 * 512)

        index_onsets = np.minimum(
            convo_offsets - half_window - 1,
            np.maximum(
                convo_onsets + half_window + 1,
                np.round_(onsets, 0, onsets) + lag_amount,
            ),
        )

        # index_onsets = np.round_(onsets, 0, onsets) + lag_amount

        # subtracting 1 from starts to account for 0-indexing
        starts = index_onsets - half_window - 1
        stops = index_onsets + half_window

        # vec = brain_signal[np.array(
        #     [np.arange(*item) for item in zip(starts, stops)])]

        for i, (start, stop) in enumerate(zip(starts, stops)):
            Y1[i, lag] = np.mean(brain_signal[start:stop].reshape(-1))

    return Y1


def build_XY(args, datum, brain_signal):
    """[summary]

    Args:
        args ([type]): [description]
        datum ([type]): [description]
        brain_signal ([type]): [description]

    Returns:
        [type]: [description]
    """
    X = np.stack(datum.embeddings).astype("float64")

    word_onsets = datum.adjusted_onset.values
    convo_onsets = datum.convo_onset.values
    convo_offsets = datum.convo_offset.values

    lags = np.array(args.lags)
    brain_signal = brain_signal.reshape(-1, 1)

    Y = build_Y(
        word_onsets,
        convo_onsets,
        convo_offsets,
        brain_signal,
        lags,
        args.window_size,
    )

    return X, Y


# def encode_lags_numba(args, X, Y, folds, fold_num, lag):
#     """[summary]
#     Args:
#         X ([type]): [description]
#         Y ([type]): [description]
#     Returns:
#         [type]: [description]
#     """
#     if args.shuffle:
#         np.random.shuffle(Y)

#     PY_hat = cv_lm_003(X, Y, folds, fold_num, lag)
#     rp, _, _ = encColCorr(Y, PY_hat)

#     return rp


def encoding_mp_prod_comp(args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, lag):
    if args.shuffle:
        np.random.shuffle(Ytra)
        np.random.shuffle(Ytes)

    PY_hat, Y_new = cv_lm_003_prod_comp(
        args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, lag
    )
    rp, _, _ = encColCorr(Y_new, PY_hat)

    return rp


# def run_save_permutation_pr(args, prod_X, prod_Y, filename):
#     """[summary]
#     Args:
#         args ([type]): [description]
#         prod_X ([type]): [description]
#         prod_Y ([type]): [description]
#         filename ([type]): [description]
#     """
#     if prod_X.shape[0]:
#         perm_rc = encode_lags_numba(args, prod_X, prod_Y)
#     else:
#         perm_rc = None

#     return perm_rc


def run_regression(args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes):
    perm_prod = []
    for i in range(args.npermutations):
        result = encoding_mp_prod_comp(
            args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, -1
        )
        if args.model_mod and "best-lag" in args.model_mod:
            best_lag = np.argmax(np.array(result))
            print("switch to best-lag: " + str(best_lag))
            perm_prod.append(
                encoding_mp_prod_comp(
                    args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, best_lag
                )
            )
        else:
            perm_prod.append(result)

    return perm_prod


# def run_save_permutation(args, prod_X, prod_Y, folds, fold_num, filename):
#     """[summary]

#     Args:
#         args ([type]): [description]
#         prod_X ([type]): [description]
#         prod_Y ([type]): [description]
#         filename ([type]): [description]
#     """
#     if prod_X.shape[0]:
#         if args.parallel:
#             print(f"Running {args.npermutations} in parallel")
#             with Pool(16) as pool:
#                 perm_prod = pool.map(
#                     partial(
#                         encode_lags_numba,
#                         args=args,
#                         prod_X=prod_X,
#                         prod_Y=prod_Y,
#                         folds=folds,
#                         fold_num=fold_num,
#                     ),
#                     range(args.npermutations),
#                 )
#         else:
#             perm_prod = []
#             for i in range(args.npermutations):
#                 result = encode_lags_numba(args, prod_X, prod_Y, folds, fold_num, -1)
#                 if args.model_mod:
#                     assert "best-lag" in args.model_mod, f"Model modification Error"
#                     best_lag = np.argmax(np.array(result))
#                     print("switch to best-lag: " + str(best_lag))
#                     perm_prod.append(
#                         encode_lags_numba(
#                             args, prod_X, prod_Y, folds, fold_num, best_lag
#                         )
#                     )
#                 else:
#                     perm_prod.append(result)
#         with open(filename, "w") as csvfile:
#             print("writing file")
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerows(perm_prod)


def load_header(conversation_dir, subject_id):
    """[summary]

    Args:
        conversation_dir ([type]): [description]
        subject_id (string): Subject ID

    Returns:
        list: labels
    """
    misc_dir = os.path.join(conversation_dir, subject_id, "misc")
    header_file = os.path.join(misc_dir, subject_id + "_header.mat")
    if not os.path.exists(header_file):
        print(f"[WARN] no header found in {misc_dir}")
        return
    header = mat73.loadmat(header_file)
    labels = header.header.label

    return labels


# def encoding_regression_pr(args, datum, elec_signal, name):
#     """[summary]
#     Args:
#         args (Namespace): Command-line inputs and other configuration
#         sid (str): Subject ID
#         datum (DataFrame): ['word', 'onset', 'offset', 'speaker', 'accuracy']
#         elec_signal (numpy.ndarray): of shape (num_samples, 1)
#         name (str): electrode name
#     """
#     # Build design matrices
#     X, Y = build_XY(args, datum, elec_signal)

#     # Split into production and comprehension
#     prod_X = X[datum.speaker == "Speaker1", :]
#     comp_X = X[datum.speaker != "Speaker1", :]

#     prod_Y = Y[datum.speaker == "Speaker1", :]
#     comp_Y = Y[datum.speaker != "Speaker1", :]

#     # Run permutation and save results
#     prod_corr = run_save_permutation_pr(args, prod_X, prod_Y, None)
#     comp_corr = run_save_permutation_pr(args, comp_X, comp_Y, None)

#     return (prod_corr, comp_corr)


def get_groupkfolds(datum, X, Y, fold_num=10):
    fold_cat = np.zeros(datum.shape[0])
    grpkfold = GroupKFold(n_splits=fold_num)
    folds = [t[1] for t in grpkfold.split(X, Y, groups=datum["conversation_id"])]

    for i in range(0, len(folds)):
        for row in folds[i]:
            fold_cat[row] = i  # turns into fold category

    fold_cat_prod = fold_cat[datum.speaker == "Speaker1"]
    fold_cat_comp = fold_cat[datum.speaker != "Speaker1"]

    return (fold_cat_prod, fold_cat_comp)


def get_kfolds(X, fold_num=10):
    print("Using kfolds")
    skf = KFold(n_splits=fold_num, shuffle=False)
    folds = [t[1] for t in skf.split(np.arange(X.shape[0]))]
    fold_cat = np.zeros(X.shape[0])
    for i in range(0, len(folds)):
        for row in folds[i]:
            fold_cat[row] = i  # turns into fold category
    return fold_cat


def write_encoding_results(args, cor_results, elec_name, mode):
    """Write output into csv files

    Args:
        args (namespace): commandline arguments
        cor_results: correlation results
        elec_name: electrode name as a substring of filename
        mode: 'prod' or 'comp'

    Returns:
        None
    """
    trial_str = append_jobid_to_string(args, mode)
    filename = os.path.join(args.full_output_dir, elec_name + trial_str + ".csv")

    with open(filename, "w") as csvfile:
        print("writing file")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(cor_results)

    return None


# def encoding_regression(args, datum, elec_signal, name):

#     # Build design matrices
#     X, Y = build_XY(args, datum, elec_signal)

#     # Get folds
#     fold_cat_prod, fold_cat_comp = get_folds(args, datum, X, Y)
#     fold_num = 5

#     # Split into production and comprehension
#     prod_X = X[datum.speaker == "Speaker1", :]
#     comp_X = X[datum.speaker != "Speaker1", :]

#     prod_Y = Y[datum.speaker == "Speaker1", :]
#     comp_Y = Y[datum.speaker != "Speaker1", :]

#     print(f"{args.sid} {name} Prod: {len(prod_X)} Comp: {len(comp_X)}")

#     # Run permutation and save results
#     trial_str = append_jobid_to_string(args, "prod")
#     filename = os.path.join(args.full_output_dir, name + trial_str + ".csv")
#     run_save_permutation(args, prod_X, prod_Y, fold_cat_prod, fold_num, filename)

#     trial_str = append_jobid_to_string(args, "comp")
#     filename = os.path.join(args.full_output_dir, name + trial_str + ".csv")
#     run_save_permutation(args, comp_X, comp_Y, fold_cat_comp, fold_num, filename)

#     return


def append_jobid_to_string(args, speech_str):
    """Adds job id to the output eletrode.csv file.

    Args:
        args (Namespace): Contains all commandline agruments
        speech_str (string): Production (prod)/Comprehension (comp)

    Returns:
        string: concatenated string
    """
    speech_str = "_" + speech_str

    if args.job_id:
        trial_str = "_".join([speech_str, f"{args.job_id:02d}"])
    else:
        trial_str = speech_str

    return trial_str
