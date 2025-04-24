import streamlit as st
import scanpy as sc
import scanpy.external as sce
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------- CACHE LOADER ----------
@st.cache_resource
def load_data():
    return sc.read("pancreas_data.h5ad")

adata = load_data()

# ---------- CHECK FUNCTIONS ----------
def is_umap_computed(adata):
    return 'X_umap' in adata.obsm

def is_pca_computed(adata):
    return 'X_pca' in adata.obsm

def is_highly_variable_computed(adata):
    return 'highly_variable' in adata.var

def has_qc_columns(adata):
    return all(k in adata.obs.columns for k in ['n_genes_by_counts', 'total_counts'])

def is_harmony_computed(adata):
    return 'X_pca_harmony' in adata.obsm

# ---------- SIDEBAR ----------
st.sidebar.title("Ρυθμίσεις")

# --- Preprocessing ---
st.sidebar.header("Preprocessing")
min_genes = st.sidebar.slider("Min genes per cell", 100, 1000, 600)
min_cells = st.sidebar.slider("Min cells per gene", 1, 10, 3)

if st.sidebar.button("Εκτέλεση Preprocessing"):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[:, [g for g in adata.var_names if not str(g).startswith(('ERCC', 'MT-', 'mt-'))]]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    adata.obs["n_genes_by_counts"] = (adata.X > 0).sum(1)
    adata.obs["total_counts"] = adata.X.sum(1)

    st.success("Preprocessing ολοκληρώθηκε.")

# --- Differential Expression ---
st.sidebar.header("Differential Expression")
run_de = st.sidebar.button("Εκτέλεση Ανάλυσης Δ.Ε.")

groupby_option = st.sidebar.selectbox(
    "Group by:", options=adata.obs_keys(),
    index=adata.obs_keys().index("disease") if "disease" in adata.obs_keys() else 0
)

# ---------- MAIN APP ----------
st.title("Διαδραστική Εφαρμογή Ανάλυσης scRNA-seq")

# Tabs
tab1, tab2, tab3 = st.tabs(["Εξερεύνηση Δεδομένων", "Διαφορική Έκφραση", "Harmony Integration"])

# ------------------ TAB 1 ------------------
with tab1:
    st.header("Εξερεύνηση Δεδομένων")

    plot_type = st.selectbox(
        "Διάλεξε plot:",
        [
            "UMAP ανά χαρακτηριστικό",
            "PCA Plot",
            "Highly Variable Genes",
            "Violin Plot (QC)",
            "Bar Plot: Celltype vs Disease"
        ]
    )

    color_by = st.selectbox(
        "Χρωματισμός με βάση:",
        options=adata.obs_keys(),
        index=adata.obs_keys().index("celltype") if "celltype" in adata.obs_keys() else 0
    )

    if plot_type == "UMAP ανά χαρακτηριστικό":
        if is_umap_computed(adata):
            sc.pl.umap(adata, color=color_by, show=True)
            st.pyplot()
        else:
            st.warning("UMAP δεν έχει υπολογιστεί.")

    elif plot_type == "PCA Plot":
        if is_pca_computed(adata):
            sc.pl.pca(adata, color=color_by, show=True)
            st.pyplot()
        else:
            st.warning("PCA δεν έχει υπολογιστεί.")

    elif plot_type == "Highly Variable Genes":
        if is_highly_variable_computed(adata):
            sc.pl.highly_variable_genes(adata, show=True)
            st.pyplot()
        else:
            st.warning("Δεν έχουν υπολογιστεί τα highly variable genes.")

    elif plot_type == "Violin Plot (QC)":
        if has_qc_columns(adata):
            sc.pl.violin(adata, keys=['n_genes_by_counts', 'total_counts'], groupby=color_by, jitter=0.4, show=True)
            st.pyplot()
        else:
            st.warning("QC δεδομένα δεν υπάρχουν ακόμα. Τρέξε preprocessing.")

    elif plot_type == "Bar Plot: Celltype vs Disease":
        if "celltype" in adata.obs and "disease" in adata.obs:
            df = pd.crosstab(adata.obs["celltype"], adata.obs["disease"])
            st.bar_chart(df)
        else:
            st.warning("Λείπουν τα πεδία 'celltype' ή 'disease'.")

# ------------------ TAB 2 ------------------
with tab2:
    st.header("Ανάλυση Διαφορικής Έκφρασης")

    if run_de:
        if groupby_option not in adata.obs:
            st.error(f"Το πεδίο '{groupby_option}' δεν υπάρχει.")
        else:
            try:
                sc.tl.rank_genes_groups(
                    adata,
                    groupby=groupby_option,
                    method="wilcoxon",
                    use_raw=False
                )
                st.success("Η ανάλυση διαφορικής έκφρασης ολοκληρώθηκε!")
            except Exception as e:
                st.error(f"Σφάλμα: {e}")

    if "rank_genes_groups" in adata.uns:
        deg_result = adata.uns["rank_genes_groups"]
        top_group = deg_result["names"].dtype.names[0]

        degs_df = pd.DataFrame({
            "gene": deg_result["names"][top_group],
            "pval": deg_result["pvals"][top_group],
            "logFC": deg_result["logfoldchanges"][top_group]
        })

        degs_df["-log10(pval)"] = -np.log10(degs_df["pval"])
        degs_df["status"] = "NS"
        degs_df.loc[(degs_df["logFC"] > 1) & (degs_df["pval"] < 0.05), "status"] = "UP"
        degs_df.loc[(degs_df["logFC"] < -1) & (degs_df["pval"] < 0.05), "status"] = "DOWN"

        st.subheader("Volcano Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=degs_df, x="logFC", y="-log10(pval)", hue="status",
                        palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"},
                        alpha=0.7)
        ax.axhline(-np.log10(0.05), color="gray", linestyle="--")
        ax.axvline(1, color="gray", linestyle="--")
        ax.axvline(-1, color="gray", linestyle="--")
        ax.set_title("Volcano Plot")
        st.pyplot(fig)

        st.subheader("Heatmap κορυφαίων γονιδίων")
        sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, show=True)
        st.pyplot()

        st.subheader("Πίνακας DEGs")
        st.dataframe(degs_df)
    else:
        st.info("Δεν έχει γίνει ακόμα ανάλυση διαφορικής έκφρασης.")

# ------------------ TAB 3 ------------------
with tab3:
    st.header("Harmony Integration")

    st.markdown("""
    Χρησιμοποιούμε τη μέθοδο Harmony για διόρθωση batch effect.
    """)

    if st.button("Εκτέλεση Harmony Integration"):
        try:
            sce.pp.harmony_integrate(adata, key='batch')
            sc.pp.neighbors(adata, use_rep="X_pca_harmony")
            sc.tl.umap(adata)
            st.success("Harmony ολοκληρώθηκε!")
        except Exception as e:
            st.error(f"Σφάλμα: {e}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("UMAP πριν το Harmony")
        if is_umap_computed(adata):
            sc.pl.umap(adata, color="batch", show=True)
            st.pyplot()
        else:
            st.info("Δεν υπάρχει ακόμα υπολογισμένο UMAP.")

    with col2:
        st.subheader("UMAP μετά το Harmony")
        if is_harmony_computed(adata):
            sc.pl.umap(adata, color="batch", show=True)
            st.pyplot()
        else:
            st.info("Δεν έχει εφαρμοστεί ακόμα το Harmony.")