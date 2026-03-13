import pandas as pd
import os

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)

    train_df.drop(columns=["difficulty_level"], inplace=True, errors="ignore")
    test_df.drop(columns=["difficulty_level"], inplace=True, errors="ignore")

    print(f"[DataLoader] Train shape : {train_df.shape}")
    print(f"[DataLoader] Test  shape : {test_df.shape}")
    print(f"[DataLoader] Train label distribution:\n{train_df['label'].value_counts().head(10)}\n")

    return train_df, test_df
