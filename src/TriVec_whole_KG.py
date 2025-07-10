# -*- coding: utf-8 -*-

import os
import itertools
import gzip
import numpy as np
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from libkge.embedding import TransE, DistMult, ComplEx, TriModel, DistMult_MCL, ComplEx_MCL, TriModel_MCL
from libkge import KgDataset
from libkge.metrics.classification import auc_roc, auc_pr
from libkge.metrics.ranking import precision_at_k, average_precision
from libkge.metrics.classification import auc_pr, auc_roc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-r", "--rule", type=str)
parser.add_argument("-e", "--extra", action="store_true")
parser.add_argument("-f", "--filtered", action="store_true")
parser.add_argument("-c", "--cut", action="store_true")
parser.add_argument("-o", "--original", action="store_true")
args = parser.parse_args()

def main():
    seed = 1234
    nb_epochs_then_check = None
    data_name = "pse"
    kg_dp_path = "../data/"
    ddis_only_path = "../../Converters/step_train_data_only/"
    input_data_path = "../../Converters/step_two_completed_train_only"
    input_data_file = f"{args.rule}{'_filtered' if args.filtered else ''}{'_extra' if args.extra else ''}_on_{args.dataset}.txt"
    data_used = "KG_whole"
    data_type = "completed"
    completed_by = args.rule
    filter = "filtered" if args.filtered else "unfiltered"
    extra = "with_extra" if args.extra else "no_extra"

    np.random.seed(seed)

    se_map_raw = [l.strip().split("\t") for l in open(os.path.join(kg_dp_path, "se_maps.txt")).readlines()]
    se_mapping = {k: v for k, v in se_map_raw} # SE code: SE  real name

    print("Importing dataset files ... ")
    benchmark_train_fd = open(os.path.join(ddis_only_path, "DDI_decagon.txt"), "rt") 
    if args.original:
        benchmark_train_full_fd = open("../../Converters/step_train_data_only/KG_whole.txt", "rt") # original Decagon KG
    else:
        benchmark_train_full_fd = open(os.path.join(input_data_path, input_data_file), "rt") # whole KG
    benchmark_valid_fd = open(os.path.join(kg_dp_path, "DDI_valid.txt"), "rt")
    benchmark_test_fd = open(os.path.join(kg_dp_path, "DDI_test.txt"), "rt")
    
    
    benchmark_train = np.array([l.strip().split() for l in benchmark_train_fd.readlines()]) # [drug1, se, drug2]
    benchmark_train_full = np.array([l.strip().split() for l in benchmark_train_full_fd.readlines()]) # whole KG
    benchmark_valid = np.array([l.strip().split() for l in benchmark_valid_fd.readlines()])
    benchmark_test = np.array([l.strip().split() for l in benchmark_test_fd.readlines()])

    benchmark_triples = np.array([[d1, se, d2] for d1, se, d2 in
                                  np.concatenate([benchmark_train, benchmark_valid, benchmark_test])]) # whole DDI.txt (only DDI)

    pse_drugs = list(set(list(np.concatenate([benchmark_triples[:, 0], benchmark_triples[:, 2]])))) # all unique drugs (only drugs)
    pse_list = set(list(benchmark_triples[:, 1])) # all unique side effects (only between different drugs)

    drug_combinations = np.array([[d1, d2] for d1, d2 in list(itertools.product(pse_drugs, pse_drugs)) if d1 != d2]) # array of cartesian product of all drug pairs \{a,a}

    print("Processing dataset files to generate a knowledge graph ... ")
    # delete raw polypharmacy data
    del benchmark_triples
    dataset = KgDataset(name=data_name)
    dataset.load_triples(benchmark_train_full, tag="bench_train")
    dataset.load_triples(benchmark_valid, tag="bench_valid")
    dataset.load_triples(benchmark_test, tag="bench_test")

    del benchmark_train_full
    del benchmark_valid
    del benchmark_test

    nb_entities = dataset.get_ents_count()
    nb_relations = dataset.get_rels_count()
    pse_indices = dataset.get_rel_indices(list(pse_list)) # indices of all unique side effects

    #all 
    d1 = np.array(dataset.get_ent_indices(list(drug_combinations[:, 0]))).reshape([-1, 1])
    d2 = np.array(dataset.get_ent_indices(list(drug_combinations[:, 1]))).reshape([-1, 1])
    drug_combinations = np.concatenate([d1, d2], axis=1) # array of cartesian product of all drug index pairs \{i,i}
    del d1
    del d2

    # grouping side effect information by the side effect type
    train_data = dataset.data["bench_train"]
    valid_data = dataset.data["bench_valid"]
    test_data = dataset.data["bench_test"]

    bench_idx_data = np.concatenate([train_data, valid_data, test_data]) # all triples in the dataset
    se_facts_full_dict = {se: set() for se in pse_indices} # empty set for all side effect index

    for s, p, o in bench_idx_data:
        if s != o and p in pse_indices:  # check if relation index is a side effect index
            se_facts_full_dict[p].add((s, p, o)) # se_i index: (drug1, se_i, drug2)

    print("Initializing the knowledge graph embedding model... ")
    # model pipeline definition
    model = TriModel(seed=seed, verbose=2)
    pipe_model = Pipeline([('kge_model', model)])

    # set model parameters
    model_params = {
        'kge_model__em_size': 100,
        'kge_model__lr': 0.01,
        'kge_model__optimiser': "AMSgrad",
        'kge_model__log_interval': 10,
        'kge_model__nb_epochs': 100,
        'kge_model__nb_negs': 6,
        'kge_model__batch_size': 5000,
        'kge_model__initialiser': 'xavier_uniform',
        'kge_model__nb_ents': nb_entities,
        'kge_model__nb_rels': nb_relations
    }

    # add parameters to the model then call fit method
    pipe_model.set_params(**model_params)

    print("Training ... ")
    pipe_model.fit(X=train_data, y=None)

    metrics_per_se = {se_idx: {"ap": .0, "auc-roc": .0, "auc-pr": .0, "p@50": .0} for se_idx in pse_indices}

    se_ap_list = []
    se_auc_roc_list = []
    se_auc_pr_list = []
    se_p50_list = []

    print("================================================================================")
    for se in tqdm(pse_indices, desc="Evaluating test data for each side-effect"):
        se_name = dataset.get_rel_labels([se])[0]
        se_all_facts_set = se_facts_full_dict[se]
        se_test_facts_pos = np.array([[s, p, o] for s, p, o in test_data if p == se])
        se_test_facts_pos_size = len(se_test_facts_pos)
        
        print(f"\nProcessing side effect: {se_name}")
        print(f"Number of positive test facts: {se_test_facts_pos_size}")
        
        if se_test_facts_pos_size == 0:
            print(f"Skipping {se_name} as there are no positive test facts")
            continue

        se_test_facts_neg = np.array([[d1, se, d2] for d1, d2 in drug_combinations
                                      if (d1, se, d2) not in se_all_facts_set
                                      and (d2, se, d1) not in se_all_facts_set])
        
        print(f"Number of negative test facts before sampling: {len(se_test_facts_neg)}")
        
        if len(se_test_facts_neg) == 0:
            print(f"Skipping {se_name} as there are no negative test facts")
            continue

        # shuffle and keep negatives with size equal to positive instances so positive to negative ratio is 1:1
        np.random.shuffle(se_test_facts_neg)
        se_test_facts_neg = se_test_facts_neg[:se_test_facts_pos_size, :]
        
        print(f"Final shapes - pos: {se_test_facts_pos.shape}, neg: {se_test_facts_neg.shape}")

        set_test_facts_all = np.concatenate([se_test_facts_pos, se_test_facts_neg])
        se_test_facts_labels = np.concatenate([np.ones([len(se_test_facts_pos)]), np.zeros([len(se_test_facts_neg)])])
        se_test_facts_scores = model.predict(set_test_facts_all)

        se_ap = average_precision(se_test_facts_labels, se_test_facts_scores)
        se_p50 = precision_at_k(se_test_facts_labels, se_test_facts_scores, k=50)
        se_auc_pr = auc_pr(se_test_facts_labels, se_test_facts_scores)
        se_auc_roc = auc_roc(se_test_facts_labels, se_test_facts_scores)

        se_ap_list.append(se_ap)
        se_auc_roc_list.append(se_auc_roc)
        se_auc_pr_list.append(se_auc_pr)
        se_p50_list.append(se_p50)

        se_code = se_name.replace("se_", "")
        metrics_per_se[se] = {"ap": se_ap, "auc-roc": se_auc_roc, "auc-pr": se_auc_pr, "p@50": se_p50}
        print("AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - P@50: %1.4f > %s: %s" %
              (se_ap, se_auc_roc, se_auc_pr, se_p50, se_code, se_mapping[se_code]), flush=True)

    se_ap_list_avg = np.average(se_ap_list)
    se_auc_roc_list_avg = np.average(se_auc_roc_list)
    se_auc_pr_list_avg = np.average(se_auc_pr_list)
    se_p50_list_avg = np.average(se_p50_list)

    print("================================================================================")
    print("[AVERAGE] AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - P@50: %1.4f" %
          (se_ap_list_avg, se_auc_roc_list_avg, se_auc_pr_list_avg, se_p50_list_avg), flush=True)
    print("================================================================================")

    if args.original:
        with open("../../Analysis/incomplete_with_proper_parameters.txt", 'a') as ff:
            ff.write(f"{se_ap_list_avg:.4f}\t{se_auc_roc_list_avg:.4f}\t{se_auc_pr_list_avg:.4f}\t{se_p50_list_avg:.4f}\t{data_used}\t{data_type}\t")
            ff.write("KG_whole\toriginal\n")
    else:
        with open(f"../results/step_two_results.txt", "a") as ff:
            ff.write(f"{se_ap_list_avg:.4f}\t{se_auc_roc_list_avg:.4f}\t{se_auc_pr_list_avg:.4f}\t{se_p50_list_avg:.4f}\t{data_used}\t{data_type}\t")
            ff.write(f"{completed_by}\t{filter}\t{extra}\n")

if __name__ == '__main__':
    main()

