"""
Text Classification Pipeline (Cleaned & Commented using ChatGPT5)

- Supports: Bag-of-Words DNN, LSTM variants, Logistic Regression, Random Forest
- Tokenization: TF/Keras Tokenizer for both text and labels
- Evaluation: accuracy + precision/recall/F1
- Outputs: saved model + CSV predictions + time/metrics ledger

NOTE: This script assumes `DNN_Functions.py`, `hyperparameters.py`, `Grab_gSheets_Functions.py` and optional
      GloVe helpers (load_glove) exist and expose the same symbols you used.

Assumes you have also downloaded the relevant datasets.  
"""

import os
import re
import time
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting if you add plots later

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Consistent TF/Keras imports (avoid mixing keras.* and tensorflow.keras.*)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

# Callbacks
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# ---------------------------------------------------------------------
# Local modules (must be present in the same environment)
# ---------------------------------------------------------------------
from Grab_gSheets_Functions import get_google_sheet, gsheet_to_df  # noqa: F401 (left for parity)
from DNN_Functions import *  # noqa: F403 (Bag_of_Words_DNN, LSTM_DNN, GLV_LSTM, GLV_FC, load_glove)
from hyperparameters import *  # noqa: F403 (expects: vocab_size, oov_tok, max_length, etc.)

# ---------------------------------------------------------------------
# User-Editable Flags (kept in-place to match your templating system)
# ---------------------------------------------------------------------
# FLAGSTART#
# THESE HAVE TO REMAIN IN POSITION AFTER THE FLAG START
TRAINING_RANGES =  []  # e.g. ["train_part1", "train_part2"]
PREDICTION_RANGES = []  # e.g. ["holdout_oct"]
MODEL_NAME = ""         # e.g. "my_text_model_v1"
ANALYSIS_METHOD = ""    # one of {"FC","LSTM","GLV-LSTM","GLV-FC","LR","RFC"}
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def set_reproducibility(seed: int = 42) -> None:
    """Set seeds for reproducible runs (best-effort)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_experiment_name() -> str:
    """Derive a stable experiment name even if __file__ is not defined."""
    try:
        return os.path.basename(__file__)
    except NameError:
        return "interactive_session"


def read_csv_concat(stems: List[str], folder: str = "Datasets") -> pd.DataFrame:
    """Read one or more CSVs from a folder and concatenate."""
    if not stems:
        raise ValueError("No dataset ranges provided.")
    dfs = []
    for stem in stems:
        path = os.path.join(folder, f"{stem}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


def duplicate_minority_rows_top(df: pd.DataFrame,
                                label_col: str = "Coded",
                                min_count: int = 2) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Move rows with < min_count for the label to the top (keeps original order otherwise).
    This preserves your original behavior which duplicates *ordering*, not rows.
    """
    merged = df.reset_index(drop=True)
    low_idx = merged.groupby(label_col).filter(lambda x: len(x) < min_count).index.tolist()
    merged = pd.concat([merged.iloc[low_idx, :], merged], ignore_index=True)
    articles = merged["Label"]
    labels = merged[label_col]
    return articles, labels, merged


def stratified_split(articles: pd.Series,
                     labels: pd.Series,
                     training_portion: float,
                     seed: int = 0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified split into train/validation based on labels."""
    if not (0 < training_portion < 1):
        raise ValueError("training_portion must be in (0,1).")
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=1 - training_portion,
                                 random_state=seed)
    (train_idx, val_idx), = sss.split(X=articles, y=labels)
    return (articles.iloc[train_idx],
            labels.iloc[train_idx],
            articles.iloc[val_idx],
            labels.iloc[val_idx])


def build_tokenizers(articles: pd.Series,
                     labels: pd.Series,
                     vocab_size: int,
                     oov_tok: str) -> Tuple[Tokenizer, Tokenizer, Dict[int, str]]:
    """Fit tokenizers for text and labels; return reverse-word index for decoding."""
    text_tok = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    text_tok.fit_on_texts(articles)
    word_index = text_tok.word_index
    reverse_word_index = {v: k for k, v in word_index.items()}

    # Label tokenizer: preserve case, split on '#', keep OOV
    label_tok = Tokenizer(num_words=None,
                          filters="#",
                          lower=False,
                          split="#",
                          char_level=False,
                          oov_token=oov_tok)
    label_tok.fit_on_texts(labels)

    return text_tok, label_tok, reverse_word_index


def decode_article_bow(vec: np.ndarray, reverse_word_index: Dict[int, str]) -> str:
    """Decode a BoW vector to tokens (non-zero indices)."""
    idxs = np.argwhere(vec > 0).ravel()
    # Keras Tokenizer texts_to_matrix() is 0-based for word index + first column for OOV.
    # We map indices back only if present in reverse_word_index.
    return " ".join([reverse_word_index.get(i, "?") for i in idxs])


def decode_article_lstm(seq: np.ndarray, reverse_word_index: Dict[int, str]) -> str:
    """Decode an LSTM sequence of token ids back to tokens."""
    return " ".join([reverse_word_index.get(i, "?") for i in seq])


def prepare_features(analysis_method: str,
                     text_tok: Tokenizer,
                     train_articles: pd.Series,
                     val_articles: pd.Series,
                     max_length: int,
                     padding_type: str,
                     trunc_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare X_train, X_val based on chosen analysis method:
    - LSTM / GLV-LSTM: padded sequences
    - others: Bag-of-Words matrix
    """
    is_lstm = ("LSTM" in analysis_method)

    if is_lstm:
        train_seq = text_tok.texts_to_sequences(train_articles)
        val_seq = text_tok.texts_to_sequences(val_articles)

        X_train = pad_sequences(train_seq, maxlen=max_length,
                                padding=padding_type, truncating=trunc_type)
        X_val = pad_sequences(val_seq, maxlen=max_length,
                              padding=padding_type, truncating=trunc_type)
    else:
        X_train = text_tok.texts_to_matrix(train_articles)
        X_val = text_tok.texts_to_matrix(val_articles)

    return X_train, X_val


def labels_to_indices(label_tok: Tokenizer, labels: pd.Series) -> np.ndarray:
    """
    Map string labels to integer indices using label tokenizer.
    Note: texts_to_sequences returns lists; for single-token labels this is [[idx], ...].
    We flatten to shape (n,) for sklearn; Keras can accept (n,1) or one-hot later.
    """
    seq = label_tok.texts_to_sequences(labels)
    arr = np.array([s[0] if len(s) > 0 else 0 for s in seq], dtype=int)
    return arr


def build_model_dispatch(analysis_method: str,
                         X_train_shape_1: int,
                         nClasses: int,
                         node: int,
                         nLayers: int,
                         vocab_size: int,
                         embedding_dim: int,
                         max_length: int,
                         tokenizer_word_index: Optional[Dict[str, int]] = None):
    """
    Factory dispatcher for model creation based on ANALYSIS_METHOD.
    Uses DNN_Functions symbols imported via wildcard.
    """
    if "FC" in analysis_method:
        model = Bag_of_Words_DNN(
            shape=X_train_shape_1,
            nClasses=nClasses,
            dropout=dropout,
            node=node,
            nLayers=nLayers
        )
        return model

    if "LSTM" in analysis_method and "GLV" not in analysis_method:
        model = LSTM_DNN(
            shape=X_train_shape_1,
            nClasses=nClasses,
            dropout=0.25,
            node=node,
            nLayers=nLayers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_length=max_length
        )
        return model

    if "GLV-LSTM" in analysis_method:
        if tokenizer_word_index is None:
            raise ValueError("tokenizer_word_index is required for GLV-LSTM.")
        print("Loading GloVe embeddings...")
        glove_embed_matrix, words_not_found = load_glove(tokenizer_word_index)  # noqa: F405

        model = GLV_LSTM(  # noqa: F405
            embed_matrix=glove_embed_matrix,
            nClasses=nClasses,
            dropout=0.25,
            node=node,
            nLayers=nLayers,
            max_length=max_length
        )
        return model

    if "GLV-FC" in analysis_method:
        if tokenizer_word_index is None:
            raise ValueError("tokenizer_word_index is required for GLV-FC.")
        print("Loading GloVe embeddings...")
        glove_embed_matrix, words_not_found = load_glove(tokenizer_word_index)  # noqa: F405

        # For GLV-FC, input dim equals vocab_size (matrix features)
        model = GLV_FC(  # noqa: F405
            embed_matrix=glove_embed_matrix,
            nClasses=nClasses,
            dropout=0.25,
            node=node,
            nLayers=nLayers,
            vocab_size=vocab_size,
            max_length=vocab_size  # aligns with FC expectations
        )
        return model

    if analysis_method == "LR":
        return LogisticRegressionCV(
            cv=2,
            penalty="l2",
            random_state=0,
            multi_class="multinomial",
            class_weight="balanced",
            solver="saga",
            verbose=2,
            max_iter=1000,
            n_jobs=-1
        )

    if analysis_method == "RFC":
        return RandomForestClassifier(
            n_estimators=1000,
            criterion="gini",
            random_state=0,
            class_weight="balanced",
            verbose=2,
            n_jobs=-1
        )

    raise ValueError(f"Unknown ANALYSIS_METHOD: {analysis_method}")


def train_model(analysis_method: str,
                model,
                X_train: np.ndarray,
                y_train_idx: np.ndarray,
                X_val: np.ndarray,
                y_val_idx: np.ndarray,
                log_root: str,
                exp_name: str) -> Dict[str, str]:
    """
    Train model depending on framework (Keras vs. Sklearn).
    Returns a dict of training summary metrics for the timekeeper.
    """
    metrics_out = {
        "Epochs": "0",
        "acc": "0",
        "loss": "0",
        "val_loss": "0",
        "val_acc": "0"
    }

    ensure_dir("Experiment_Outputs")
    ensure_dir(log_root)

    start = time.time()

    if analysis_method in {"LR", "RFC"}:
        # Sklearn expects 1D y
        y_train = y_train_idx.ravel()
        y_val = y_val_idx.ravel()
        X_data = np.vstack((X_train, X_val))
        y_data = np.concatenate((y_train, y_val))
        model.fit(X_data, y_data)

        filename = os.path.join(
            "Experiment_Outputs",
            f"{exp_name}{datetime.now().strftime('%Y%m%d-%H%M%S')}model_{analysis_method}.sav"
        )
        joblib.dump(model, filename)
    else:
        # Keras
        logdir = os.path.join(log_root, datetime.now().strftime("%Y%m%d-%H%M%S") + exp_name)
        tb = TensorBoard(log_dir=logdir)

        # Keras can accept integer indices if the model uses sparse categorical loss
        history = model.fit(
            X_train, y_train_idx,
            epochs=num_epochs,                  # from hyperparameters.py
            batch_size=batch_size,              # from hyperparameters.py
            validation_data=(X_val, y_val_idx),
            verbose=2,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, min_delta=0.01, restore_best_weights=True),
                tb
            ]
        )

        # Save Keras model
        model.save(
            os.path.join(
                "Experiment_Outputs",
                f"{exp_name}{datetime.now().strftime('%Y%m%d-%H%M%S')}model_{analysis_method}.h5"
            )
        )

        # Collect final metrics (modern TF uses 'accuracy' not 'acc')
        hist = history.history
        metrics_out["Epochs"] = str(len(history.epoch))
        metrics_out["acc"] = f"{hist.get('accuracy', [0])[-1]:.6f}"
        metrics_out["loss"] = f"{hist.get('loss', [0])[-1]:.6f}"
        metrics_out["val_loss"] = f"{hist.get('val_loss', [0])[-1]:.6f}"
        metrics_out["val_acc"] = f"{hist.get('val_accuracy', [0])[-1]:.6f}"

    elapsed = timedelta(seconds=(time.time() - start))
    metrics_out["TimeTaken_ModelCreation"] = str(elapsed)
    return metrics_out


def predict_and_evaluate(analysis_method: str,
                         model,
                         text_tok: Tokenizer,
                         label_tok: Tokenizer,
                         prediction_df: pd.DataFrame,
                         max_length: int,
                         padding_type: str,
                         trunc_type: str,
                         exp_name: str,
                         model_name: str) -> Dict[str, float]:
    """
    Run predictions on PredictionData and compute metrics.
    Saves per-row predictions to CSV for Keras models (probabilities available).
    """
    txt = prediction_df["Label"]

    if ("LSTM" in analysis_method):
        seq = text_tok.texts_to_sequences(txt)
        X_test = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    else:
        X_test = text_tok.texts_to_matrix(txt)

    # Predict
    if analysis_method in {"LR", "RFC"}:
        # Sklearn .predict returns class indices if trained on ints
        pred_idx = model.predict(X_test)

        # label_tok.index_word maps 1-based indices to label strings
        # Our pred_idx are 1..N; map carefully (0 is OOV if it ever appears)
        idx_to_label = label_tok.index_word  # e.g. {1:'A', 2:'B', ...}
        model_estimations = np.array([idx_to_label.get(int(i), oov_tok) for i in pred_idx])
    else:
        # Keras returns logits/probs -> argmax
        preds = model.predict(X_test, verbose=0)
        pred_idx = np.argmax(preds, axis=1)

        # Keras label tokenizer indices start at 1; subtract 0 only if your model produced 1..N.
        # Here argmax produces 0..(n_classes-1), but your label_tok is 1..N.
        # So we *add 1* to map back to tokenizer indices.
        mapped_idx = pred_idx + 1
        idx_to_label = label_tok.index_word
        model_estimations = np.array([idx_to_label.get(int(i), oov_tok) for i in mapped_idx])

        # Save per-row output with probabilities
        out = pd.DataFrame({
            "Variable": prediction_df.get("Variable", pd.Series(index=prediction_df.index, dtype=object)),
            "Label": prediction_df["Label"],
            "PredictedClass": model_estimations,
            "PredictedProbability": np.max(preds, axis=1)
        })
        out_path = os.path.join(
            "Experiment_Outputs",
            f"{exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}Prediction_Output_{model_name}.csv"
        )
        out.to_csv(out_path, index=False)

    # Evaluate (true labels are strings in PredictionData['Coded'])
    true_labels = prediction_df["Coded"]
    report = metrics.classification_report(true_labels, model_estimations)
    acc = metrics.accuracy_score(true_labels, model_estimations)
    prf1 = metrics.precision_recall_fscore_support(true_labels, model_estimations, average="weighted")

    print("-" * 25)
    print(f"Classification Report for {analysis_method}")
    print(report)
    print(f"\nAccuracy: {acc:.4f}")
    print("-" * 25)

    return {
        "TestAccuracy": float(acc),
        "TestPrecision": float(prf1[0]),
        "TestRecall": float(prf1[1]),
        "TestF1": float(prf1[2]),
    }


def write_timekeeper(path: str, timekeeper: Dict[str, str]) -> None:
    """Write timekeeper stats to a CSV (key,value rows)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for k, v in timekeeper.items():
            f.write(f"{k}, {v}\n")


def main():
    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    set_reproducibility(42)

    exp_name = get_experiment_name()
    start_wall = time.time()

    TimeKeeper: Dict[str, str] = dict()
    TimeKeeper["ExperimentName"] = exp_name
    TimeKeeper["TimeStart"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TimeKeeper["ModelName"] = MODEL_NAME
    TimeKeeper["Epochs"] = "0"
    TimeKeeper["acc"] = "0"
    TimeKeeper["loss"] = "0"
    TimeKeeper["val_loss"] = "0"
    TimeKeeper["val_acc"] = "0"

    print("\n" + "-" * 60)
    print("Begin Preparing Data")
    print("-" * 60)
    print(f"Training Datasets  : {TRAINING_RANGES}")
    print(f"Prediction Datasets: {PREDICTION_RANGES}")
    print("-" * 60)

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    training_df = read_csv_concat(TRAINING_RANGES)
    prediction_df = read_csv_concat(PREDICTION_RANGES)

    print(f"Training Data Size   : {training_df.shape}")
    print(f"Prediction Data Size : {prediction_df.shape}")
    print("-" * 60)
    print("Head of Prediction Data:")
    print(prediction_df.head())
    print("-" * 60)

    # -----------------------------------------------------------------
    # Balance/minority handling and splits
    # -----------------------------------------------------------------
    articles, labels, merged = duplicate_minority_rows_top(training_df, label_col="Coded", min_count=2)

    # `training_portion` expected from hyperparameters.py
    train_articles, train_labels, val_articles, val_labels = stratified_split(
        articles=articles,
        labels=labels,
        training_portion=training_portion,  # noqa: F405
        seed=0
    )

    print(f"Training X count: {len(train_articles)}")
    print(f"Training y count: {len(train_labels)}")
    print(f"Validation X count: {len(val_articles)}")
    print(f"Validation y count: {len(val_labels)}")

    # -----------------------------------------------------------------
    # Tokenizers
    # -----------------------------------------------------------------
    text_tok, label_tok, reverse_word_index = build_tokenizers(
        articles=articles,
        labels=labels,
        vocab_size=vocab_size,  # noqa: F405
        oov_tok=oov_tok         # noqa: F405
    )

    # Label indices for training/validation
    y_train_idx = labels_to_indices(label_tok, train_labels)
    y_val_idx = labels_to_indices(label_tok, val_labels)

    # -----------------------------------------------------------------
    # Features
    # -----------------------------------------------------------------
    feat_start = time.time()
    X_train, X_val = prepare_features(
        analysis_method=ANALYSIS_METHOD,
        text_tok=text_tok,
        train_articles=train_articles,
        val_articles=val_articles,
        max_length=max_length,     # noqa: F405
        padding_type=padding_type, # noqa: F405
        trunc_type=trunc_type      # noqa: F405
    )

    TimeKeeper["TimeTaken_Preparation"] = str(timedelta(seconds=(time.time() - feat_start)))

    # -----------------------------------------------------------------
    # Model build & train
    # -----------------------------------------------------------------
    model = build_model_dispatch(
        analysis_method=ANALYSIS_METHOD,
        X_train_shape_1=X_train.shape[1],
        nClasses=nClasses,             # noqa: F405
        node=node,                     # noqa: F405
        nLayers=nLayers,               # noqa: F405
        vocab_size=vocab_size,         # noqa: F405
        embedding_dim=embedding_dim,   # noqa: F405
        max_length=max_length,         # noqa: F405
        tokenizer_word_index=text_tok.word_index
    )

    train_metrics = train_model(
        analysis_method=ANALYSIS_METHOD,
        model=model,
        X_train=X_train,
        y_train_idx=y_train_idx.reshape(-1, 1) if ANALYSIS_METHOD not in {"LR", "RFC"} else y_train_idx,
        X_val=X_val,
        y_val_idx=y_val_idx.reshape(-1, 1) if ANALYSIS_METHOD not in {"LR", "RFC"} else y_val_idx,
        log_root="logs/scalars",
        exp_name=exp_name
    )

    # Merge training metrics to TimeKeeper
    TimeKeeper.update(train_metrics)

    # -----------------------------------------------------------------
    # Predict & Evaluate
    # -----------------------------------------------------------------
    eval_metrics = predict_and_evaluate(
        analysis_method=ANALYSIS_METHOD,
        model=model,
        text_tok=text_tok,
        label_tok=label_tok,
        prediction_df=prediction_df,
        max_length=max_length,     # noqa: F405
        padding_type=padding_type, # noqa: F405
        trunc_type=trunc_type,     # noqa: F405
        exp_name=exp_name,
        model_name=MODEL_NAME
    )

    TimeKeeper.update({k: f"{v}" for k, v in eval_metrics.items()})

    # -----------------------------------------------------------------
    # Finish & persist time ledger
    # -----------------------------------------------------------------
    total_elapsed = timedelta(seconds=(time.time() - start_wall))
    TimeKeeper["TimeTaken_Script"] = str(total_elapsed)
    TimeKeeper["TimeFinished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    time_path = os.path.join(
        "Experiment_Outputs",
        f"{exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}stats_timekeeper.csv"
    )
    write_timekeeper(time_path, TimeKeeper)

    print("\nDone.")


if __name__ == "__main__":
    main()
