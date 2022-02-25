from collections import Counter, defaultdict
import scanpy as sc
import pandas as pd


def preproc_anndata_hypercluster(anndata_obj):
    # Make object copy
    ann_data_working = anndata_obj.copy()

    # Process with scanpy
    sc.pp.filter_cells(ann_data_working, min_genes=200)
    sc.pp.filter_genes(ann_data_working, min_cells=3)
    sc.pp.normalize_total(ann_data_working, target_sum=1e4)
    sc.pp.log1p(ann_data_working)
    sc.pp.highly_variable_genes(
        ann_data_working, min_mean=0.0125, max_mean=3, min_disp=0.5
    )
    sc.pp.scale(ann_data_working, max_value=10)
    sc.tl.pca(ann_data_working, svd_solver="arpack")
    sc.pp.neighbors(ann_data_working, n_neighbors=10, n_pcs=40)

    return ann_data_working


def hypercluster_anndata(
    anndata_obj,
    original_clustering_name,
    min_cells_recluster=500,
    subcluster_resolution=1,
    min_new_cluster_size=0,
    verbose=True,
    return_anndata=False,
    type="leiden",
    do_preproc=True,
):
    """Generate a hyperclustering of an AnnData object

    :param anndata_obj: AnnData object
    :param original_clustering_name: name of clustering column in .obs to hypercluster, cluster partitions will be maintained
    :param min_cells_recluster: minimum number of cells in a cluster to consider for reclustering
    :param subcluster_resolution: resolution parameter to be passed to clustering function, smaller values give more clusters
    :param min_new_cluster_size: cutoff size for keeping new clusters
    :param verbose: verbosity
    :param return_anndata: flag specifying if the processed anndata object should be returned
    :param type: clustering algorithm to use 'leiden' or 'louvain'
    :param do_preproc: flag specifying if the anndata object should be preprocessed
    """

    if do_preproc:
        ann_data_working = preproc_anndata_hypercluster(anndata_obj)
    else:
        ann_data_working = anndata_obj.copy()

    # Identify which clusters to subcluster
    ct_counter = Counter(ann_data_working.obs[original_clustering_name])
    cell_types_identical = list(
        k for k in ct_counter if ct_counter[k] <= min_cells_recluster
    )
    cell_types_recluster = list(
        k for k in ct_counter if ct_counter[k] > min_cells_recluster
    )

    # Dictionary for the new mapping
    new_cell_mapping = {}

    cluster_map = {}

    if type == "leiden":
        restricted_name = "leiden_R"
    elif type == "louvain":
        restricted_name = "louvain_R"
    else:
        raise Exception("Unknown clustering type")

    # Recluster other clusters
    for ct in cell_types_recluster:
        if verbose:
            print(f"Reclustering {ct}...")

        original_cell_names = list(
            ann_data_working.obs.loc[
                ann_data_working.obs[original_clustering_name] == ct
            ].index
        )

        if type == "leiden":
            sc.tl.leiden(
                ann_data_working,
                restrict_to=(original_clustering_name, (ct,)),
                resolution=subcluster_resolution,
            )
        elif type == "louvain":
            sc.tl.louvain(
                ann_data_working,
                restrict_to=(original_clustering_name, (ct,)),
                resolution=subcluster_resolution,
            )

        new_cluster_names = set(
            list(ann_data_working.obs.loc[original_cell_names][restricted_name])
        )
        if verbose:
            print(f"\t{len(new_cluster_names)} subclusters identified")

        for nct in new_cluster_names:
            cells_in_new_cluster = list(
                ann_data_working.obs[ann_data_working.obs[restricted_name] == nct].index
            )
            new_cell_mapping[nct] = cells_in_new_cluster

            cluster_map[nct] = ct

    # Keep new clusters only over a specified size
    new_clusters_keep = list(
        k for k in new_cell_mapping if len(new_cell_mapping[k]) >= min_new_cluster_size
    )
    discarded_clusters = list(
        k for k in new_cell_mapping if len(new_cell_mapping[k]) < min_new_cluster_size
    )

    # Copy clusters that are too small without reclustering
    for ct in cell_types_identical:
        if verbose:
            print(f"Keeping {ct}...")
        sel = ann_data_working.obs[original_clustering_name] == ct
        cells_in_celltype = list(
            ann_data_working.obs[original_clustering_name].index[sel]
        )
        new_cell_mapping[ct] = cells_in_celltype

        cluster_map[ct] = ct

    if verbose:
        print(
            f"Keeping {len(new_clusters_keep)}/{len(new_clusters_keep) + len(discarded_clusters)} clusters"
        )

    # Generate dataframe for the new clusters
    cell_id_column_name = "cell_name"
    hypercluster_columns_name = "hypercluster"

    cn_new = []
    cluster_new = []

    for cl_keep in new_clusters_keep:
        new_cluster_dict = new_cell_mapping[cl_keep]
        for cn in new_cluster_dict:
            cn_new.append(cn)
            cluster_new.append(cl_keep)

    new_cluster_df = pd.DataFrame(
        {
            cell_id_column_name: cn_new,
            hypercluster_columns_name: pd.Categorical(cluster_new),
        }
    ).set_index(cell_id_column_name)

    # TODO: (Optional) trim cluster_map to only contain clusters retained

    if return_anndata:
        ann_data_working.obs = ann_data_working.obs.join(new_cluster_df)
        return {
            "cluster_map": cluster_map,
            "new_clusters": new_cluster_df,
            "ann_data_working": ann_data_working,
        }
    else:
        return {"cluster_map": cluster_map, "new_clusters": new_cluster_df}
