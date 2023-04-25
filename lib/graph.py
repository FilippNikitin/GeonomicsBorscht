import pandas as pd
import torch

from torch_geometric.data import Data


def read_hic_graph(meth_path, hic_path, resolution=50000):
    column_names = ["chr", "start_pos", "end_pos", "bin_id",
                    "methylated_read_count", "total_read_count"]
    cell_data = pd.read_csv(meth_path, sep="\t", header=None, names=column_names)
    cell_data["bin_id"] = cell_data["bin_id"].str[3:].astype(int)
    cell_data["methylated_read_count"] = pd.to_numeric(cell_data["methylated_read_count"],
                                                       errors="coerce")
    cell_data["total_read_count"] = pd.to_numeric(cell_data["total_read_count"],
                                                  errors="coerce")
    cell_data.fillna(0, inplace=True)
    column_names = ["num1", "left_chr", "left_pos", "num2", "num3",
                    "right_chr", "right_pos", "contact"]
    hic_data = pd.read_csv(hic_path, sep="\t", header=None, names=column_names)
    hic_data["left_start_pos"] = hic_data["left_pos"] // resolution * resolution
    hic_data["right_start_pos"] = hic_data["right_pos"] // resolution * resolution
    for part in ["left_", "right_"]:
        hic_data = pd.merge(cell_data.add_prefix(part), hic_data,
                            on=[f'{part}start_pos', f'{part}chr'],
                            how='inner')
        hic_data.drop([f"{part}start_pos", f"{part}end_pos", f"{part}pos", f"{part}chr"], axis=1,
                      inplace=True)
    groups = hic_data.groupby(by=["left_bin_id", "right_bin_id"]).agg({"contact": "sum"})
    cell_data = cell_data[["bin_id", "methylated_read_count", "total_read_count"]].set_index(
        "bin_id")
    return groups, cell_data


def join_graphs(graph_list):
    res_hic, res_cell = graph_list[0]
    for i in graph_list[1:]:
        res_hic.add(i[0], fill_value=0)
        res_cell.add(i[1], fill_value=0)
    return res_hic, res_cell


def get_pyg_graph(hic, cell):
    edge_index = hic.index
    edge_index = [[i[0] for i in edge_index],
                  [i[1] for i in edge_index]]
    edge_index = torch.tensor(edge_index).long()
    edge_attr = torch.tensor(hic["contact"].values)

    met = torch.tensor(cell["methylated_read_count"].values.astype(float))
    tot = torch.tensor(cell["total_read_count"].values.astype(float))
    met[tot != 0] = met[tot != 0] / tot[tot != 0]
    x = torch.arange(len(met)).view(-1, 1)
    graph = Data(edge_index=edge_index,
                 edge_attr=edge_attr,
                 x=x, y=met.float())
    return graph


def read_dataframe(df, hic_path, sc_path, resolution=50000):
    graphs = {}
    for i in pd.unique(df.iloc[:, -1]):
        selected = df[df.iloc[:, -1] == i]
        data = []
        for hic, sc in zip(selected.iloc[:, -0], selected.iloc[:, 1]):
            data.append(read_hic_graph(f"{sc_path}/{sc}", f"{hic_path}/{hic}", resolution))
        g = join_graphs(data)
        graphs[i] = get_pyg_graph(*g)
    return graphs


if __name__ == "__main__":
    hic_path = "../GSM4736679_191216-CEMBA-mm-P56-CEMBA191126-9J-1-CEMBA191126-9J-2-A10_ad010_hic.txt"
    met_path = "../GSM4736679_allc_191216-CEMBA-mm-P56-CEMBA191126-9J-1-CEMBA191126-9J-2-A10_ad010.tsv.gz_cg_final"
    hic, cell = read_hic_graph(met_path, hic_path)
    hic, cell = join_graphs([[hic, cell], [hic, cell]])
    pyg = get_pyg_graph(hic, cell)
    print(pyg)

