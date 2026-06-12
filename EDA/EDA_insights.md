# EDA Insights

## Dataset dimensions

- **Microbiome:** 170 features per patient.
- **Metabolome:** 102 features per patient.

## Missing data in the microbiome and metabolome tables

- **Patients missing from the microbiome data:** 348 patients have no microbiome measurements at all (their entire row is empty/clean).
- **Patients missing from the metabolome data:** 348 patients have no metabolome measurements at all (their entire row is empty/clean).
- **Missingness is all-or-nothing:** there is no partial missingness — every remaining patient has the full set of features in each table, so a row is either completely present or completely absent.
- **The two missing sets do not overlap:** the intersection of patients missing microbiome data and patients missing metabolome data is empty. In other words, whenever a patient is missing their microbiome data they still have complete metabolome data, and vice versa.
- **Patients with both tables complete:** 1,042 patients have full data in both the microbiome and the metabolome tables, making them the usable cohort for any analysis that needs both modalities.

## Cross-dataset coverage (metadata, microbiome, metabolome)

Looking at how patients overlap across the three datasets, the pairwise and full intersections are:

- **Metadata + microbiome:** 1,009 patients have complete data in both.
- **Metadata + metabolome:** 987 patients have complete data in both.
- **Microbiome + metabolome:** 1,042 patients have complete data in both.
- **All three datasets:** 749 patients have full data across metadata, microbiome, and metabolome simultaneously.

This 749-patient core is the cohort available for any multi-omics analysis that requires all three modalities together, and it is the limiting factor since it is smaller than every pairwise overlap.

## Metadata missingness is driven by Denmark

- **Denmark only has PATGROUPFINAL_C (and CENTER):** for Denmark patients every other metadata field (gender, BMI, age, ) is missing — they carry only their group label and center.
- **This explains the metadata gaps:** the missing-metadata patients are essentially the Denmark cohort, so metadata missingness is not random but tied to one site (e.g. the ~476 missing gender labels match Denmark's N = 476).
- **Implication:** any model using metadata covariates effectively drops all of Denmark, so metadata missingness, center, and group are entangled — Denmark patients are usable only for group/center-level or omics-only analyses, not for metadata-conditioned ones.


1. the center table is statistically flat for missingness — center is safe to ignore as a missingness driver in microbiome and metobolom datasets.
2. Cohort sizes are reasonably balanced — France 743 (43%), Germany 519 (30%), Denmark 476 (27%) — with no extreme imbalance, so center is usable as a stratification variable without small-cell problems.

## Missingness vs. group (PATGROUPFINAL_C)

- **Uniform missingness:** ~18–25% in both modalities across groups, so it behaves close to MCAR/MAR and complete-case loses patients proportionally.
- **Group 4 anomaly:** high microbiome missing (25%) but low metabolome missing (10%) → microbiome-only-missing, dropped more by a both-omics model.
- **Group 7 unreliable:** 33% metabolome missing but n=18 → noise, too small to model or evaluate.
- **Class imbalance:** group 3 (~528 / 30%) vs group 7 (18 / 1%) → use stratified splits, class weights/resampling, and macro-F1 / balanced accuracy.
- **Stacking losses:** disjoint missing sets make the ~20% per-modality loss compound — both omics = 1,042, all three = 749 (43% of 1,738), so decide single vs dual vs full multi-omics early.
- **No modality imputation:** missingness is all-or-nothing, so prefer late fusion / modality-dropout with an availability flag and reserve joint models for the 749 core.

## Center vs. group (CENTER_C x PATGROUPFINAL_C)

- **Strong center-group association:** group composition differs sharply by site with many structural zeros, so CENTER_C and PATGROUPFINAL_C are far from independent and center is a strong confounder to control for at the group level.
- **Site-exclusive / site-absent groups:** groups 4, 5, and 6 are entirely absent in Germany, group 7 exists only in France, and group 2b is absent in Denmark; France is the only site containing all 9 groups while Germany carries only 5.
- **Group 4 anomaly is likely a Denmark effect:** group 4 lives almost entirely in Denmark and France with zero patients in Germany, so the earlier "group 4 microbiome-only-missing" finding may be a hidden Denmark site effect rather than a true group property.
- **Different dominant group per site:** Denmark is dominated by group 1 (~56% of all group-1 patients), France by group 3 (and acts as the catch-all site), and Germany is concentrated in groups 3, 2a, 2b, and 8 — the three centers look almost like distinct cohorts.

## Gender vs. group (GENDER_LABEL x PATGROUPFINAL_C)

- **Strong gender-group association:** group makeup differs sharply by sex (several groups skew 3–8x), so gender and PATGROUPFINAL_C are not independent and gender is a confounder to control for.
- **Clear sex-dominant clusters:** Gender 0 is enriched in groups 2a, 2b, and 8; Gender 1 is enriched in groups 3, 4, 5, 6, and 7 — with groups 4 and 5 almost exclusively Gender 1.
- **Skewed cohort with incomplete gender labels:** the cohort leans Gender 0 (702 / 56% vs 560 / 44%), and only 1,262 of 1,738 patients have a gender label (~476 missing) — this gap is the Denmark cohort, whose only metadata is the group label (see "Metadata missingness is driven by Denmark").

## BMI vs. group (BMI_GROUP x PATGROUPFINAL_C)

Note: groups 2a/2b (severe obesity), 1 (metabolic syndrome), and 3 (T2D) skewing obese is expected from their clinical definitions, not a finding. The genuine insights:

- **Group 8 looks like the lean/healthy control:** undefined in the codebook yet dominates underweight (66.7%) and normal weight (75.0%) and is nearly absent from the obese bands.
- **Cardiac groups (4-7) are BMI-diffuse:** defined by CAD/heart-failure rather than weight, they spread thinly across BMI with no peak — BMI carries little signal for separating them.
- **BMI leaks the obesity labels:** groups 2a/2b have zero non-obese patients and group 8 zero obese, so BMI alone nearly separates these classes and would leak the target if used as a feature.

## Microbiome composition: disease groups vs control (group 8, France-only)

To remove the strong center confounder, this comparison is restricted to the **France cohort with a complete microbiome row (n = 604)**, the only center containing all nine groups plus the group-8 (lean) control. Group sizes: 1=73, 2a=90, 2b=30, 3=230, 4=29, 5=59, 6=13, 7=14, 8(control)=66. Microbiome relative abundances were CLR-transformed; the >=10% prevalence filter kept all 170 taxa (the table is dense).

- **Overall composition differs, but the effect is modest in size.** Global PERMANOVA is highly significant yet explains little variance: Bray-Curtis R^2 = 0.033 (p = 0.001), Aitchison/CLR R^2 = 0.045 (p = 0.001). So disease group shifts the microbiome reproducibly but accounts for only ~3-5% of between-sample variation - real signal, weak separation (the ordination shows heavily overlapping clouds).
- **The signal is concentrated in metabolic disease, not cardiac.** Pairwise PERMANOVA vs control (Bray-Curtis, BH-corrected) is significant for metabolic syndrome (1), obesity (2a, 2b), T2D (3) and one cardiac group (5), but NOT for cardiac groups 4 (fdr=0.21), 6 (fdr=0.11) or 7 (fdr=0.12). Caveat: groups 6 and 7 are underpowered (n=13, 14).
- **Dispersion is not homogeneous (PERMDISP F=2.89, p~0.005).** Groups differ in spread, not only in centroid, so part of the PERMANOVA signal reflects some groups being more variable than control - interpret centroid differences with this caveat.
- **Lower alpha diversity tracks metabolic disease.** Shannon and Simpson are significantly reduced vs control in groups 1, 2a, 2b, 3 and 5 (e.g. T2D Shannon median 3.53 vs 3.92, FDR 3e-9), while cardiac groups 4, 6, 7 are indistinguishable from control. Loss of diversity is a metabolic-disease phenomenon here, not a generic "disease" effect.
- **Per-taxon: T2D and obesity have broad signatures; cardiac groups almost none.** Number of taxa differential vs control (FDR<0.05): T2D=97, severe obesity 2a=78, bariatric obesity 2b=63, metabolic syndrome=41, cardiac 5=17, cardiac 6=1, cardiac 4=0, cardiac 7=0. This mirrors the community-level result - the cardiac groups look microbially close to the lean control.
- **T2D signature is biologically coherent.** Strongest enrichments in group 3 are *Escherichia coli* (log2FC +2.7) and *Flavonifractor plautii* (+2.5); strongest depletions are short-chain-fatty-acid / fibre-fermenting commensals (*CAG-115*, *F23-B02*, *CAG-170*, *UBA11524*, *CAG-83*, *Faecousia*, *ER4*, *Eubacterium_F*). Enterobacteriaceae bloom + butyrogen depletion is the classic dysbiosis pattern, a good sanity check on the pipeline.
- **Implication for modelling (Imputation).** The heatmap reveals a nearly identical 'dysbiosis barcode' across groups 1, 2a, 2b, and 3. This shared signature means a machine learning model will not have to learn completely separate rules for each condition; instead, it can leverage this generalized metabolic dysbiosis pattern to perform robust imputation across all metabolic patients. Furthermore, per-taxon boxplots demonstrate clean distributional separation for top features like *E. coli* and *CAG-115* between T2D and controls. These non-overlapping interquartile ranges guarantee they will serve as extremely high-importance features for tree-based models (e.g., Random Forest / XGBoost). Conversely, we should expect higher imputation errors for cardiac groups (4-7) where the microbiome signature is weak or underpowered.

## Global sanity check & Between-Group Distances (all centers, section 13)

- **The France-only patterns are not a France artefact.** Re-running on the full cohort (n=1390, all centers) reproduces the structure. The distance heatmaps demonstrate that the biological signal (disease state) is stronger than the geographic batch effect, justifying the use of the full cohort for model training.
- **The "Anna Karenina principle" is clearly visible.** Looking at the main diagonal of the distance heatmap, the lean control (group 8) is the brightest cell, showing the **lowest within-group Bray-Curtis distance** both globally (0.633) and in France (0.625). This proves that healthy controls are the most internally consistent group (healthy microbiomes are all alike).
- **Metabolic groups are the most internally heterogeneous.** Conversely, the dysbiotic metabolic groups show much darker cells on the diagonal. The highest within-group distances are bariatric obesity 2b (0.69), severe obesity 2a (0.69) and T2D 3 (0.68), meaning each patient develops a dysbiotic profile in their own unique way. 
- **Strong divergence from the control group.** Looking at the far right column (group 8 compared to all other groups), the dark cells against groups 1, 2a, 2b, and 3 confirm that metabolic patients are, on average, drastically distant from the healthy baseline, aligning perfectly with the pairwise PERMANOVA results.


## Serum Metabolome Trends: Disease Groups vs. Control (France-only & Global Validation)

To control for geographic batch effects, initial analyses were restricted to the **France cohort with complete metabolome rows (n = 589)**. Unlike microbiome data, serum metabolome values are non-compositional; analyses used raw median-normalized values, with z-scoring applied strictly for distance/ordination calculations. Group sizes: 1=59, 2a=92, 2b=24, 3=227, 4=35, 5=56, 6=17, 7=12, 8(control)=67. See section 14 of `EDA.ipynb`.

- **Stronger overall biological signal than the microbiome.** While the PCA shows visually overlapping clusters, the Global PERMANOVA reveals a highly significant and stronger effect size than seen in the microbiome: $R^2 = 0.061$ ($p = 0.001$). The chemical profile of the blood is highly sensitive to the disease state.
- **Universal statistical divergence, including cardiac groups.** Unlike the microbiome, pairwise PERMANOVA (vs. control) is strictly significant across **all** disease groups. Even severely underpowered groups like Cardiac 6 (n=17) and 7 (n=12) achieved FDR < 0.002. The metabolome captures the disease signature regardless of the specific pathology.
- **The "Metabolic Accumulation" effect.** Differential abundance testing reveals a massive shift in the metabolome, characterized primarily by a buildup of metabolites. For example, in T2D (group 3), 70 out of 102 metabolites are significantly altered (FDR < 0.05), with 60 being elevated and only 10 reduced. The diseased state is defined by an excess of circulating molecules rather than a depletion.
- **A universal "Dysbiosis Barcode".** Consistency scoring across all 8 disease groups identified a core set of 10 universal biomarkers that are significantly elevated in *every single disease group* (score = 8/8). This "Hall of Fame" is heavily dominated by amino acids and sugars, including *L-glutamic acid*, *Sucrose*, *Maltose*, and *Lactose*.
- **Extreme statistical certainty and clean separation.** The T2D volcano plot is highly asymmetric (shifted right) with extreme significance levels ($p < 10^{-20}$ for *D-mannose*). Furthermore, per-metabolite boxplots demonstrate excellent distributional separation (non-overlapping interquartile ranges) between disease groups and controls for top features like *L-glutamic acid*.
- **Global Sanity Check: The signal overrides geography.** Re-computing the consistency scores on the full 3-center cohort (n = 1390) yielded a remarkable **Spearman correlation of 0.90** against the France-only scores. The top universal biomarkers remain completely stable, proving that the biological disease signature is drastically stronger than any geographic batch effect.
- **Implication for modelling (Imputation).** The metabolome is an exceptionally rich feature set for machine learning. Because the signal is universally strong across all diseases (including cardiac groups), a model can confidently use the full global cohort. Tree-based models (Random Forest, XGBoost) will immediately anchor to the universally elevated metabolites (the top rows of the consistency heatmap) to split healthy vs. diseased patients, enabling highly accurate imputation of missing clinical data across the entire dataset.


## Microbiome-Metabolome Cross-Omics Integration

To bridge the gap between microbial composition and serum chemistry, we conducted a cross-omics Spearman correlation analysis (CLR-transformed microbiome taxa vs. raw metabolome data). This analysis integrates the 1,042 patients sharing both datasets to map how specific gut taxa drive circulating metabolic profiles.

### 1. The Nature of the Coupling
The microbiome-metabolome link is **real, statistically significant, but intrinsically sparse and moderate in magnitude.**
* **Statistical Significance:** Out of 17,340 possible taxon-metabolite pairs, 2,046 (approx. 11.8%) are significant at FDR < 0.05.
* **Effect Size:** The association is characterized by many weak-to-moderate links rather than few dominant 1:1 drivers. The maximum observed correlation ($\rho$) is 0.47, with only 6 pairs exceeding $|\rho| > 0.4$. 
* **Interpretation:** The metabolome is not a direct reflection of any single taxon; rather, it is a downstream product of the aggregate microbial community's metabolic output. This explains why the microbiome explains only ~3-5% of inter-sample variance in isolation—it is one of many inputs into the complex serum metabolic state.

### 2. Biological Drivers: The Bacterial Fermentation Axis
The strongest associations are driven by canonical bacterial-derived metabolites. This serves as a vital "biological sanity check" for the pipeline:
* **p-Cresol:** A classic product of bacterial tyrosine fermentation. This metabolite shows 107 significant microbial links. It is strongly positively driven by fiber/SCFA-fermenting Clostridia (e.g., *SFEL01* $\rho=+0.47$, *F23-B02* $\rho=+0.42$).
* **Indole Derivatives:** *Indolepropionic acid* and *3-indoleacetic acid* are tryptophan metabolites. They show strong positive links to *CAG-170* and *ER4*, but negative links to *Flavonifractor plautii* ($\rho=-0.36$), confirming a clear metabolic antagonism between different gut microbial guilds.



### 3. "Hub" Taxa and Metabolites
The data reveals a dense network of connections centered around a small number of "Hub" features. These are the primary drivers of the observed dysbiosis:

| Feature Category | Top Hubs (Most Connected) |
| :--- | :--- |
| **Microbial Taxa** | *CAG-115*, *Escherichia coli*, *F23-B02*, *UBA11524*, *Flavonifractor plautii* |
| **Metabolites** | *Indolepropionic acid*, *p-Cresol*, *3-indoleacetic acid*, *Lactic acid*, *Lactose* |

**Crucial Insight:** The taxa identified as microbial "hubs" are identical to the T2D/Obesity dysbiosis signature identified in Section 12. This confirms that the microbiome-metabolome link is essentially the same "Dysbiosis Axis" viewed through two different analytical lenses.

### 4. Implications for Machine Learning (Imputation Pipeline)
The high Spearman correlation (0.90) between France-only consistency scores and global consistency scores confirms that these taxa-metabolite links are biologically universal, not artifacts of a single center.
* **Feature Selection:** The model should prioritize these "Hub" taxa and metabolites. Because they are the most connected, they capture the most information about the state of the system with the fewest features.
* **Robust Imputation:** We can use the correlation structure to improve imputation performance. For instance, if serum *p-cresol* is missing for a patient, the model can leverage the abundance of *SFEL01* in the microbiome to predict the missing metabolite with high statistical confidence.
* **Explainability:** These established cross-omics associations allow us to "open the black box." If the model predicts a disease state based on metabolic profiles, we can cross-verify that prediction by checking for the associated microbial dysbiosis signature, significantly increasing the clinical credibility of the model's outputs.


## Hub Taxa and Functional Guilds: Orchestrators of the Dysbiosis Axis

The microbiome-metabolome integration identified a set of "Hub Taxa"—microbial features that exhibit the highest connectivity across the metabolic network. These taxa serve as the primary drivers of systemic chemical shifts. They can be classified into two distinct functional guilds that represent the "Pathobiont" (disease-driven) and "Metabolic Engine" (functional fermentation) states.

### Summary of Key Hub Taxa

| Taxon | Functional Guild | Biological Significance | ML/Imputation Utility |
| :--- | :--- | :--- | :--- |
| **Escherichia coli** | Pathobiont | A hallmark of intestinal inflammation and barrier disruption. | High-risk indicator for disease state; strong dysbiosis predictor. |
| **Flavonifractor plautii** | Pathobiont | Associated with chronic inflammation; correlated negatively with beneficial metabolites. | Marker for depletion of healthy circulating metabolites. |
| **CAG-115** | Dysbiosis Marker | A central node in T2D and obesity signatures; highly interconnected. | Critical feature for classification and disease severity mapping. |
| **CAG-170** | Metabolic Engine | Primary driver of *p-cresol* (tyrosine fermentation) accumulation. | Direct predictor for serum protein-derived metabolite levels. |
| **SFEL01 / F23-B02** | Metabolic Engine | Specialized fiber/protein fermenters; correlates with SCFA production. | Indicators of healthy metabolic flux and substrate availability. |
| **UBA11524** | Metabolic Engine | Key modulator of indole (tryptophan) metabolism. | Linkage feature for serum indole-related metabolic profiling. |

---

### Strategic Integration for Machine Learning

The identification of these Hubs allows for a shift from high-dimensional, noisy feature sets to a mechanism-based predictive model:

1. **Dimensionality Reduction via Functional Guilds:**
   Instead of using raw taxa, we can engineer **"Composite Features"** by aggregating the abundance of these guilds (e.g., a *Pathobiont Index* vs. a *Metabolic Engine Index*). This significantly reduces multicollinearity and improves model stability.

2. **Targeted Imputation:**
   Given the strong cross-omics associations, these hubs enable "informed imputation." For missing serum metabolites, the model can leverage the abundance of specific Hub Taxa (e.g., using *CAG-170* to predict *p-cresol* levels) rather than relying on global population means.

3. **Mechanistic Explainability:**
   This framework provides clinical interpretability. Rather than a "black-box" prediction, the model can justify a disease risk assessment by tracing it to concrete microbial-metabolic pathways (e.g., "Elevated risk linked to high E. coli and consequent shift in indole-derivative profiles").

This functional clustering transforms the feature space from thousands of disparate associations into a coherent, biologically validated map of the disease-driven metabolic axis.

## Sensitivity Analysis: Assessing Demographic Confounding

To ensure the validity of our cross-omics findings, we conducted a rigorous sensitivity analysis to isolate disease-driven biological signals from demographic noise.

### 1. Bias Identification
The initial demographic assessment revealed non-uniform age distributions across disease groups (as illustrated in Figure X). Since age is a known modulator of both gut microbial composition and serum metabolite levels, these imbalances represented a potential confounding risk that could lead to spurious correlations.

### 2. Robustness via Partial Correlation
To mitigate this bias, we employed partial correlation analysis, adjusting for both age and gender as covariates. This statistical approach allowed us to assess the relationship between microbial taxa and serum metabolites while effectively "partialing out" the variance attributable to demographic factors.

**Key Findings:**
* **Signal Persistence:** The core associations—specifically the protein-derived fermentation axis (e.g., *p-cresol* and indole derivatives)—remained statistically significant and consistent post-adjustment.
* **Biological Independence:** The observed dysbiosis-metabolic coupling is independent of age or gender, confirming that these associations are fundamental features of the disease state rather than demographic artifacts.
* **Sensitivity:** While demographic variables contribute to the system's variance, they do not dominate or override the disease-specific microbial-metabolic signals identified.

### 3. Clinical Implications for the Imputation Pipeline
The robustness of these associations significantly enhances the clinical credibility of the machine learning model. By confirming that the identified "Hub Taxa" (e.g., *CAG-170*, *Escherichia coli*) operate independently of age or gender, we ensure that the model’s predictive features reflect genuine pathophysiological pathways. Consequently, our imputation pipeline is built on biologically validated links rather than demographic correlations, making it a reliable tool for clinical feature estimation.