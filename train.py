import os.path
import pickle
from argparse import ArgumentParser

import pandas as pd
import wandb
import yaml
from lightning import Trainer
from torch_geometric.data import DataLoader

from lib.dataset import GraphDataset
from lib.graph import read_dataframe
from lib.llightning_model import LitNodePredictor

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", default=None)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--ckpt_path", default=None)

    args = parser.parse_args()

    config_dict = yaml.load(open(args.config), Loader=yaml.FullLoader)
    network_module = config_dict["model"]["module_name"]
    network_params = config_dict["model"]["params"]
    optimizer_module = config_dict["optimization"]["optimizer_module_name"]
    optimizer_params = config_dict["optimization"]["optimizer_params"]
    scheduler_module = config_dict["optimization"]["scheduler_module_name"]
    scheduler_params = config_dict["optimization"]["scheduler_params"]
    criterion = config_dict["criterion"]
    metrics = config_dict["metrics"]
    dataset = config_dict["datasets"]

    model = LitNodePredictor(network_module, network_params, criterion,
                             optimizer_module, optimizer_params, scheduler_module,
                             scheduler_params, metrics)
    df = pd.read_csv(dataset["clustering_file"])
    df = df.loc[:, dataset["columns"]]
    if os.path.exists(dataset["graph_path"]):
        graphs = pickle.load(open(dataset["graph_path"], "rb"))
    else:
        graphs = read_dataframe(df, dataset["hic_path"], dataset["met_path"],
                                dataset["resolution"])
        pickle.dump(graphs, open(dataset["graph_path"], "wb"))

    test_dataset = GraphDataset({i: graphs[i] for i in dataset["test_clusters"]},
                                k_hop=dataset["k_hop"], n_graphs=dataset["n_graphs"],
                                quantile=dataset["contact_quantile"])
    train_dataset = GraphDataset(
        {i: graphs[i] for i in graphs if i not in dataset["test_clusters"]},
        k_hop=dataset["k_hop"], n_graphs=dataset["n_graphs"],
        quantile=dataset["contact_quantile"])

    train_loader = DataLoader(train_dataset, batch_size=config_dict["trainer"]["batch_size"],
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=config_dict["trainer"]["batch_size"],
                            num_workers=2, pin_memory=True)
    print(next(iter(val_loader)))
    wandb.init(
        entity=config_dict["wandb"]["entity"],
        settings=wandb.Settings(start_method="fork"),
        project=config_dict["wandb"]["project"],
        name=config_dict["wandb"]["run_name"],
        config=config_dict
    )
    wandb.watch(model.network, log="all", log_freq=1000, log_graph=True)
    del config_dict["trainer"]["batch_size"]
    trainer = Trainer(**config_dict["trainer"])
    if args.test:
        trainer.test(model, ckpt_path=args.ckpt_path, dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)
