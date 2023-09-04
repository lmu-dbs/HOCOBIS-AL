import json
import os
import statistics

import click
import numpy as np

from src.data.dataset import get_handler, get_dataset, get_transform
from src.models import get_backbone, get_learner
from src.data.database import MLFlowLogger
from src.al import get_strategy
from src.utils import str_list, set_seed
from src.settings import RUN_CONFIG


@click.command()
@click.option('--seeds', default=[0, 1, 2, 3, 4], type=str_list)  # 5 seeds
@click.option('--tracking_uri', required=True, help='MLflow tracking uri.')
def main(seeds, tracking_uri):
    n_labels = [20, 30, 50, 50, 50, 50]
    acq_strategy = "RandomSampling"
    learners = ["supervised", "pseudolabel", "fixmatch", "flexmatch"]
    for ds in ["DirtyMNIST", "ImbalancedMNIST", "IntraClassImbalanceMNIST"]:
        variants = ['1']
        imbalanced = False
        if ds == "ImbalancedMNIST":
            imbalanced = True
        for _learner in learners:
            for variant in variants:
                # load config
                config_path = os.path.join(RUN_CONFIG, f'{ds.lower()}_{variant}.json')
                with open(config_path, 'r') as config_file:
                    config = json.load(config_file)

                # load dataset
                X_tr, Y_tr, X_te, Y_te, class_counts = get_dataset(ds, imbalanced=imbalanced)

                early_stopping = False
                if 'early_stopping' in config["train_params"]:
                    early_stopping = config["train_params"]["early_stopping"]

                config['ds_params']['transform'] = get_transform(ds)
                n_pool = len(Y_tr)

                # model settings
                backbone = config["backbone"]

                # evaluation loop
                # for _n in n_labels:
                backbone_args = config['backbone_args']
                settings = {
                    "strategy": acq_strategy,
                    "dataset": ds,
                    "n_initial": n_labels[0],
                    "n_final": sum(n_labels),
                    "rounds": len(n_labels)-1,
                    "learner": _learner,
                    "backbone": backbone,
                    "backbone_args": backbone_args,
                    "seed": seeds,  # save all seeds to mlflow for reproduction
                    "epochs": config["train_params"]["epochs"],
                    "early_stop": early_stopping,
                    "imbalanced": imbalanced,
                    "batch_size_tr": config["ds_params"]["loader_tr_args"]["batch_size"],
                    "batch_size_te": config["ds_params"]["loader_te_args"]["batch_size"],
                    "learning_rate": config["train_params"]["lr"],
                    "weight_decay": config["train_params"]["wd"]
                }
                #  One mlflow run over all seeds
                database = MLFlowLogger(experiment_name=f'3-UAAL-{ds}-Sev5', tracking_uri=tracking_uri)
                _, _ = database.init_experiment(hyper_parameters=settings)

                pre_learner = get_learner(learner=_learner)
                pre_backbone = get_backbone(model_architecture=backbone)
                handler = get_handler(ds)

                # n_query = int(_n/al_rounds)  # 20->4, 50->10, 100->20, 200->40
                accuracies_per_seed = {iteration: [] for iteration in range(0, len(n_labels))}
                for seed in seeds:
                    set_seed(seed)
                    print(f"-> Eval with seed={seed}")
                    # generate initial labeled pool; dependent on seed
                    idxs_lb = np.zeros(n_pool, dtype=bool)
                    idxs_tmp = np.arange(n_pool)
                    np.random.shuffle(idxs_tmp)
                    idxs_lb[idxs_tmp[:n_labels[0]]] = True
                    # size_labeled += n_query
                    learner = pre_learner(pre_backbone, backbone_args, X_tr, Y_tr, idxs_lb, config, handler)
                    strategy, strat_params = get_strategy(acq_strategy, learner=learner)

                    tr_acc, tr_loss = learner.train()
                    P = learner.predict_label(X_te, Y_te)

                    acc = np.zeros(len(n_labels))
                    acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
                    result = {f"acc_{seed}": acc[0], f"tr_loss_{seed}": tr_loss, f"tr_acc_{seed}": tr_acc}
                    accuracies_per_seed[0].append((acc[0], np.sum(idxs_lb)))

                    database.log_results(result=result, step=np.sum(idxs_lb))
                    rd = 1
                    for n_query in n_labels[1:]:
                        q_idxs = strategy.query(n_query)
                        # size_labeled += n_query
                        idxs_lb[q_idxs] = True
                        learner.update(idxs_lb)
                        tr_acc, tr_loss = learner.train()
                        P = learner.predict_label(X_te, Y_te)

                        acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
                        accuracies_per_seed[rd].append((acc[rd], np.sum(idxs_lb)))
                        result = {f"acc_{seed}": acc[rd], f"tr_loss_{seed}": tr_loss, f"tr_acc_{seed}": tr_acc}

                        database.log_results(result=result, step=np.sum(idxs_lb))
                        print('testing accuracy {}'.format(acc[rd]))
                        rd += 1

                for key, val in accuracies_per_seed.items():
                    con_accs = [tup[0] for tup in val]
                    con_samples = [tup[1] for tup in val]
                    database.log_results(result={"avg_acc": statistics.mean(con_accs)}, step=con_samples[0])
                    database.log_results(result={"var_acc": statistics.variance(con_accs)}, step=con_samples[0])
                    database.log_results(result={"std_acc": statistics.stdev(con_accs)}, step=con_samples[0])

                # database.log_results(result={"avg_acc": statistics.mean(accuracies_per_seed)})
                # database.log_results(result={"var_acc": statistics.variance(accuracies_per_seed)})
                # database.log_results(result={"std_acc": statistics.stdev(accuracies_per_seed)})
                # close experiment after all seeds have been visited
                database.finalise_experiment()


if __name__ == '__main__':
    main()
