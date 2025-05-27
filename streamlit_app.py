#Εισαγωγή της Streamlit για web διεπαφή

import streamlit as st
#Scanpy για ανάλυση scRNA-seq δεδομένων

import scanpy as sc
#Scanpy για ανάλυση scRNA-seq δεδομένων

import scanpy.external as sce
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import base64

#Συνάρτηση που ενσωματώνει εικόνα φόντου ως background στο interface

def add_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

#Κλήση της συνάρτησης
add_background("image.png")



#Φόρτωση δεδομένων

#CACHE LOADER 
@st.cache_resource
def load_data():
    return sc.read("pancreas_data.h5ad")

# Initialize session state for adata if not exists
if 'adata' not in st.session_state:
    st.session_state.adata = load_data()

adata = st.session_state.adata

#CHECK FUNCTIONS 
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

#SIDEBAR 
st.sidebar.title("Ρυθμίσεις")

#Preprocessing
st.sidebar.header("Preprocessing")
min_genes = st.sidebar.slider("Min genes per cell", 100, 1000, 600)
min_cells = st.sidebar.slider("Min cells per gene", 1, 10, 3)

if st.sidebar.button("Εκτέλεση Preprocessing"):
    # 1)Filter cells & genes
    #Φιλτράρισμα κελιών με βάση τον αριθμό εκφραζόμενων γονιδίων
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    #Φιλτράρισμα γονιδίων που εμφανίζονται σε πολύ λίγα κύτταρα
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # 2)Remove ERCC/MT genes
    adata = adata[:, [
        g for g in adata.var_names
        if not str(g).startswith(('ERCC', 'MT-', 'mt-'))
    ]]
    
    # 3)Normalize & log
    #Κανονικοποίηση έκφρασης ώστε κάθε κύτταρο να έχει ίδιο συνολικό σήμα
    sc.pp.normalize_total(adata, target_sum=1e4)

    #Λογαριθμικός μετασχηματισμός για ομαλοποίηση κατανομής
    sc.pp.log1p(adata)

    # 4)HVG & store raw
    # Εντοπισμός των πιο μεταβλητών γονιδίων (HVGs) για downstream ανάλυση
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5
    )
    adata.raw = adata
    
    # 5)Subset to HVGs, scale, PCA, neighbors, UMAP
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    # 6)QC metrics
    adata.obs["n_genes_by_counts"] = (adata.X > 0).sum(1)
    adata.obs["total_counts"] = adata.X.sum(1)
    
    # Update session state
    st.session_state.adata = adata
    st.success("Preprocessing ολοκληρώθηκε.")
    st.rerun()

#Differential Expression
st.sidebar.header("Differential Expression")
run_de = st.sidebar.button("Εκτέλεση Ανάλυσης Δ.Ε.")
groupby_option = st.sidebar.selectbox(
    "Group by:", options=adata.obs_keys(),
    index=adata.obs_keys().index("disease")
          if "disease" in adata.obs_keys() else 0
)

#MAIN APP 
st.title("Διαδραστική Εφαρμογή Ανάλυσης scRNA-seq")
tab1, tab2, tab3 = st.tabs([
    "Εξερεύνηση Δεδομένων",
    "Διαφορική Έκφραση",
    "Harmony Integration"
])

#TAB 1
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
        index=adata.obs_keys().index("celltype")
              if "celltype" in adata.obs_keys() else 0
    )

    if plot_type == "UMAP ανά χαρακτηριστικό":
        if is_umap_computed(adata):
            fig = sc.pl.umap(adata, color=color_by, return_fig=True)
            st.pyplot(fig)
        else:
            st.warning("UMAP δεν έχει υπολογιστεί.")

    elif plot_type == "PCA Plot":
        if is_pca_computed(adata):
            fig = sc.pl.pca(adata, color=color_by, return_fig=True)
            st.pyplot(fig)
        else:
            st.warning("PCA δεν έχει υπολογιστεί.")

    elif plot_type == "Highly Variable Genes":
        if is_highly_variable_computed(adata):

#Εντοπισμός των πιο μεταβλητών γονιδίων (HVGs) για downstream ανάλυση
            sc.pl.highly_variable_genes(adata, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
        else:
            st.warning("Δεν έχουν υπολογιστεί τα highly variable genes.")

    elif plot_type == "Violin Plot (QC)":
        if has_qc_columns(adata):
            # Create a much larger figure
            fig, axes = plt.subplots(2, 1, figsize=(20, 16))
            
            # Plot each metric separately for better visibility
            sc.pl.violin(
                adata,
                keys='n_genes_by_counts',
                groupby=color_by,
                jitter=0.4,
                rotation=90,  # Rotate labels 90 degrees
                ax=axes[0],
                show=False
            )
            axes[0].set_title('Number of Genes by Counts', fontsize=16)
            axes[0].tick_params(axis='x', labelsize=14)
            axes[0].tick_params(axis='y', labelsize=14)
            
            sc.pl.violin(
                adata,
                keys='total_counts',
                groupby=color_by,
                jitter=0.4,
                rotation=90,  # Rotate labels 90 degrees
                ax=axes[1],
                show=False
            )
            axes[1].set_title('Total Counts', fontsize=16)
            axes[1].tick_params(axis='x', labelsize=14)
            axes[1].tick_params(axis='y', labelsize=14)
            
            # Increase spacing between subplots
            plt.subplots_adjust(hspace=0.5)
            plt.tight_layout(pad=3.0)
            st.pyplot(fig)
        else:
            st.warning("QC δεδομένα δεν υπάρχουν ακόμα. Τρέξε preprocessing.")

    elif plot_type == "Bar Plot: Celltype vs Disease":
        if "celltype" in adata.obs and "disease" in adata.obs:
            df = pd.crosstab(
                adata.obs["celltype"],
                adata.obs["disease"]
            )
            st.bar_chart(df)
        else:
            st.warning("Λείπουν τα πεδία 'celltype' ή 'disease'.")

#TAB 2
with tab2:
    st.header("Ανάλυση Διαφορικής Έκφρασης")

    if run_de:
        if groupby_option not in adata.obs:
            st.error(f"Το πεδίο '{groupby_option}' δεν υπάρχει.")
        else:
            try:
                
                #Ανάλυση διαφορικής έκφρασης ανά ομάδα μεθόδου Wilcoxon
                sc.tl.rank_genes_groups(
                    adata,
                    groupby=groupby_option,
                    method="wilcoxon",
                    use_raw=False
                )
                st.success("Η ανάλυση διαφορικής έκφρασης ολοκληρώθηκε!")
            except Exception as e:
                st.error(f"Σφάλμα: {e}")

    #Ανάλυση διαφορικής έκφρασης ανά ομάδα μεθόδου Wilcoxon
    if "rank_genes_groups" in adata.uns:
        # 1)Flat DataFrame of DE results
        #Ανάλυση διαφορικής έκφρασης ανά ομάδα μεθόδου Wilcoxon
        de_df = sc.get.rank_genes_groups_df(adata, group=None)
        
        # 2)Choose group
        groups = de_df["group"].unique().tolist()
        selected_group = st.selectbox(
            "Επίλεξε ομάδα για Volcano Plot:", groups, index=0
        )
        
        # 3)Filter & rename
        degs_df = (
            de_df[de_df["group"] == selected_group]
            .rename(columns={
                "names": "gene",
                "pvals": "pval",
                "logfoldchanges": "logFC"
            })
            .assign(**{"-log10(pval)": lambda df: -np.log10(df["pval"])})
        )
        degs_df["status"] = "NS"
        degs_df.loc[
            (degs_df["logFC"] > 1) & (degs_df["pval"] < 0.05),
            "status"
        ] = "UP"
        degs_df.loc[
            (degs_df["logFC"] < -1) & (degs_df["pval"] < 0.05),
            "status"
        ] = "DOWN"

        #Volcano Plot
        st.subheader(f"Volcano Plot ({selected_group})")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=degs_df,
            x="logFC",
            y="-log10(pval)",
            hue="status",
            palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"},
            alpha=0.7,
            ax=ax
        )
        ax.axhline(-np.log10(0.05), linestyle="--")
        ax.axvline(1, linestyle="--")
        ax.axvline(-1, linestyle="--")
        ax.set_title(f"Volcano Plot για {selected_group}")
        st.pyplot(fig)

        #Heatmap κορυφαίων γονιδίων
        top_genes = [
            g for g in degs_df["gene"].tolist()[:10]
            if g in adata.var_names
        ]
        if not top_genes:
            st.warning("Δεν υπάρχουν διαθέσιμα κορυφαία γονίδια για heatmap.")
        else:

        #Ανάλυση διαφορικής έκφρασης ανά ομάδα μεθόδου Wilcoxon
            fig = sc.pl.rank_genes_groups_heatmap(
                adata,
                groupby=groupby_option,
                var_names=top_genes,
                use_raw=False,
                return_fig=True
            )
            st.pyplot(fig)

        #Πίνακας DEGs
        st.subheader("Πίνακας DEGs")
        st.dataframe(degs_df.reset_index(drop=True))
    else:
        st.info("Δεν έχει γίνει ακόμα ανάλυση διαφορικής έκφρασης.")

#TAB 3
with tab3:
    st.header("Harmony Integration")
    st.markdown("Χρησιμοποιούμε τη μέθοδο Harmony για διόρθωση batch effect.")

    if st.button("Εκτέλεση Harmony Integration"):
        if not is_pca_computed(adata):
            st.error("Πρέπει πρώτα να τρέξεις το Preprocessing για να υπολογιστεί το PCA!")
        elif 'batch' not in adata.obs.columns:
            st.error("Δεν βρέθηκε η στήλη 'batch' στα δεδομένα. Το Harmony χρειάζεται μια στήλη batch για να τρέξει.")
        else:
            try:
                #Εκτέλεση batch correction με Harmony integration
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
            fig = sc.pl.umap(adata, color="batch", return_fig=True)
            st.pyplot(fig)
        else:
            st.info("Δεν υπάρχει ακόμα υπολογισμένο UMAP.")
    with col2:
        st.subheader("UMAP μετά το Harmony")
        if is_harmony_computed(adata):
            fig = sc.pl.umap(adata, color="batch", return_fig=True)
            st.pyplot(fig)
        else:
            st.info("Δεν έχει εφαρμοστεί ακόμα το Harmony.")

#Κομμάτι για credits
#Εμφάνιση του κουμπιού για άνοιγμα του modal
st.markdown("""
    <style>
    .credits-button {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
    }
    </style>
    <div class="credits-button">
        <a href="#open-modal"><button style="padding: 10px 20px; border-radius: 10px;">Credits</button></a>
    </div>
""", unsafe_allow_html=True)

#modal (CSS-driven, χωρίς JS)
st.markdown("""
    <style>
    /* Βασικό modal background */
    .modal:target {
        visibility: visible;
        opacity: 1;
    }

    .modal {
        position: fixed;
        top: 0; left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.6);
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.25s ease;
        z-index: 10000;
    }

    .modal-content {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 60vw;
        max-height: 70vh;
        overflow-y: auto;
        background: black;
        padding: 2rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }

    .close {
        position: absolute;
        top: 10px;
        right: 15px;
        text-decoration: none !important;
        font-size: 48px;
        font-weight: bold;
        color: #333;
    }

    </style>

    <div id="open-modal" class="modal">
        <div class="modal-content">
            <a href="#" class="close">×</a>
            <h2>Ομάδα Ανάπτυξης</h2>
            <p>Αυτή η εφαρμογή δημιουργήθηκε στο πλαίσιο του μαθήματος Τεχνολογία Λογισμικού, του ΣΤ' Εξαμήνου με διδάσκοντες τον Επίκουρο καθηγητή κ. Βραχάτη Αριστείδη και τον Υποψήφιο Διδάκτωρα κ. Λάζαρο Κωνσταντίνο. Συνεισέφεραν οι παρακάτω φοιτητές στα αντίστοιχα κομμάτια:</p>
            <ul>
                <li><strong> Βαμβακά Αικατερίνη (Α.Μ. Π2017014):</strong> Σχεδιασμός της υλοποίησης, οπτικοποιήσεις των αποτελεσμάτων που θα παράγει η εφαρμογή, καθώς και περιγραφή του dockerization της εφαρμογής.</li>
                <li><strong> Σαραντίδης Χρήστος (Α.Μ.: inf2022182):</strong> Τίτλος, Περίληψη, Εισαγωγή καθώς και ανάλυση της υλοποίησης της εφαρμογής με τεχνικές λεπτομέρειες.</li>
                <li><strong>Σκρουμάς Βασίλειος (Α.Μ.: inf2022189):</strong> UML διαγράμματα (Class Diagram & Use Case Diagram) που θα περιγράφουν την αρχιτεκτονική και τη λειτουργικότητα της εφαρμογής.</li>
            </ul>
            <p>Η εφαρμογή αναπτύχθηκε με χρήση Python, Streamlit.</p>
            <p>Επικοινωνήστε: <a href="mailto:inf2022182@ionio.gr (Σαραντίδης Χρήστος)">inf2022182@ionio.gr</a></p>
        </div>
    </div>
""", unsafe_allow_html=True)
