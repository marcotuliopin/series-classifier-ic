import pandas as pd
import numpy as np
from keras.utils import timeseries_dataset_from_array
import tensorflow_datasets as tfds


def main(dados_info, data_group_name, database_name, colnames, seq_len, overlap):
    df, df_label = read_data(dados_info, data_group_name, database_name)
    df_label = df_label.drop(df_label.index[len(df_label) - 1])

    for tmps in df_label["timestamp_end"][:-1]:
        print(len(df[df["timestamp"] >= tmps]["timestamp"]))

    real_cp = get_real_cp(df, df_label, tmps)
    data, real_cp = get_data_from_df(df, real_cp)
    data, labels = get_samples_from_time_series(data, df_label['classe'], seq_len, overlap, True, real_cp) 

    return data, labels


def read_data(dados_info, data_group_name, database_name):
    df = pd.read_parquet("../data/01_og_HASC/{}/{}/{}_concat.parquet".format(dados_info, data_group_name, database_name))
    df_label = pd.read_csv("../data/01_og_HASC/{}/{}/{}.label".format(dados_info, data_group_name, database_name), header=None)
    df_label.columns = ["timestamp_start", "timestamp_end", "classe"]
    df.reset_index(drop=True, inplace=True)
    return df, df_label


def get_real_cp(df, df_label, tmps):
    real_cp = []
    for tmp_start, tmp_end in zip(df_label["timestamp_start"],df_label["timestamp_end"]) :
        print(tmps)
        min_tmp = max(df[(df["timestamp"] >= tmp_start) & (df["timestamp"] <= tmp_end)]["timestamp"])
        index_ch = df[df["timestamp"] == min_tmp].index.values[0]+1
        real_cp.append(index_ch)
    return real_cp
    

def get_data_from_df(df, real_cp):
    data = df.drop(columns='timestamp').apply(lambda x:np.linalg.norm(x.values), axis=1)
    data = data.drop(index=[i for i in range(real_cp[0])]).reset_index(drop=True)
    start_p = real_cp[0]
    real_cp = [cp - start_p for cp in real_cp]
    data = data.drop(index=[i for i in range(real_cp[-1], len(data))]).reset_index(drop=True)
    return data, real_cp


def get_samples_from_time_series(data, df_label, seq_len, overlap=0, is_sequence=False, real_cp=None):
    if is_sequence:
        sequences = np.split(data.to_numpy(), real_cp[1:-1])
    else:
        sequences = data.values
    labels = []
    split_sequences = []
    for seq, label in zip(sequences, df_label):
        split_sequences.append(split_times_series(seq, seq_len, overlap))
        labels += [label] * len(split_sequences[-1])
    data = pd.concat(split_sequences).reset_index(drop=True)
    labels = pd.Series(labels)
    return data, labels


def split_times_series(series, seq_len, overlap):
    dataset = timeseries_dataset_from_array(data=series, 
                                            targets=None, 
                                            sequence_length=seq_len,
                                            sampling_rate=1,
                                            sequence_stride=seq_len*(1 - overlap),
                                            batch_size=None)
    df = pd.DataFrame(tfds.as_numpy(dataset))
    return df