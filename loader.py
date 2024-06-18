import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

def load_data(data_path, noAgg=False):

    # Read edges, features and classes from csv files
    df_edges = pd.read_csv(osp.join(data_path, "edges.csv"), usecols=["clId1","clId2"])
    df_features = pd.read_csv(osp.join(data_path, "background_nodes.csv"), nrows=1000000)
    df_classes_raw1 = pd.read_csv(osp.join(data_path, "connected_components.csv"))
    df_classes_raw2 = pd.read_csv(osp.join(data_path, "nodes.csv"))
    df_classes = pd.merge(df_classes_raw1, df_classes_raw2, on='ccId')
    df_classes = df_classes.drop('ccId', axis=1)
    
    # Merge classes and features in one Dataframe
    df_class_feature = pd.merge(df_classes, df_features)

    #costruiamo il df con le edges formate solo dai nodi che conosciamo
    known_txs = df_class_feature["clId"].values
    df_edges = df_edges[(df_edges["clId1"].isin(known_txs)) & (df_edges["clId2"].isin(known_txs))]

    #rinomino le colonne delle etichette, ccLabel -> class
    df_class_feature = df_class_feature.rename(columns={'ccLabel': 'class'})

    # Build indices for features and edge types
    # Prendo i valori univoci di clId e class, e in ordine crescente gli vado a dare un indice 
    # che parte da 0. Quindi avrò la chiave che sarà il clId/class e il valore composto
    # dall'indice assegnato.
    features_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["clId"].unique()))}
    class_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["class"].unique()))}
    print("feauture_idx:\n" , features_idx) 
    print("class_idx:\n" , class_idx) 

    # Apply index encoding to features
    # Queste due istruzioni non fanno altro che sostituire il valore originale
    # di clId/class con gli indici creati precedentemente
    df_class_feature["clId"] = df_class_feature["clId"].apply(lambda name: features_idx[name])
    df_class_feature["class"] = df_class_feature["class"].apply(lambda name: class_idx[name])

    # Apply index encoding to edges
    # Qui fa la stessa identica cosa fatta al "df_class_feature" ma la applica al
    # "df_edges"
    df_edges["clId1"] = df_edges["clId1"].apply(lambda name: features_idx[name])
    df_edges["clId2"] = df_edges["clId2"].apply(lambda name: features_idx[name])

    #Inverto le colonne class e clId
    df_class_feature = df_class_feature[['clId', 'class'] + list(df_class_feature.columns[2:])]

    print(df_class_feature)
    print(df_edges)
    
    return df_class_feature, df_edges


def data_to_pyg(df_class_feature, df_edges):

    # Define PyTorch Geometric data structure with Pandas dataframe values
    edge_index = torch.tensor([df_edges["clId1"].values,
                            df_edges["clId2"].values], dtype=torch.long)
    x = torch.tensor(df_class_feature.iloc[:, 2:].values, dtype=torch.float)
    y = torch.tensor(df_class_feature["class"].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data = RandomNodeSplit(num_val=0.15, num_test=0.2)(data)

    return data

def reduce_features(df, corr_min=0.9):
    print("df shape original:", df.shape)
    corr = df[df.columns[97:]].corr()
    df_feat = corr.unstack().reset_index()
    #print("df:", df.head())
    df_feat.columns = ["f1", "f2", "value"]
    df_feat = df_feat[df_feat.f1 != df_feat.f2]
    df_feat = df_feat[(df_feat.value > corr_min) | (df_feat.value < -corr_min)]
    df_feat = df_feat.reset_index(drop=True)
    #print("df_feat: ", df_feat.head())
    to_remove = []
    existent = []
    for index, row in df_feat.iterrows():
      new = (row.f1, row.f2)
      new2 = (row.f2, row.f1)
      if (not new in existent) and (not new2 in existent):
        existent.append(new)
      else:
        to_remove.append(index)
    #print("to_remove: ", to_remove)
    df_feat = df_feat.drop(to_remove)
    df_feat = df_feat.reset_index(drop=True)
    #print(f"Pairs of aggregated features with corr > {corr_min}: {len(df_feat)}")
    col_to_remove=df_feat["f2"]
    for c in col_to_remove:
        if c in df.columns:
            df=df[df.columns[df.columns != c]]
    return df
