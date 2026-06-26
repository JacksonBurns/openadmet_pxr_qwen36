# `openadmet_pxr_qwen36`

Letting `Qwen3.6 27B` just `autoresearch` the best possible model for the [OpenADMET PXR](https://huggingface.co/spaces/openadmet/pxr-challenge) activity prediction challenge.

## Challenge Description

This is the description of the challenge data:

```
# 💊 OpenADMET PXR Blind Challenge
## Background: Why PXR Matters
Evaluating PXR liabilities is a fundamental pillar of a late-stage ADMET cascade. PXR functions as a xenobiotic sensor, detecting foreign compounds and marshalling drug-metabolizing enzymes and transporters by activating their transcription. It primarily regulates **CYP3A4**, the enzyme responsible for metabolizing approximately **50% of all marketed drugs**.
Activation of PXR can lead to:
- **Drug-Drug Interactions (DDIs):** Accelerated metabolism can reduce co-administered drug concentrations to sub-optimal levels.
- **Hepatotoxicity:** Increased production of reactive, toxic metabolites.
- **Chemoresistance:** Enhanced clearance of chemotherapeutic agents in tumor cells.
Drug discovery teams face a unique challenge with PXR due to its large, flexible ligand-binding pocket, which accommodates a wide range of chemical structures. PXR is also relatively underrepresented in the literature, with only **~800 high-quality pEC50 values** from nearly 150 papers in ChEMBL.
## The Activity Dataset
At Octant, OpenADMET has generated a PXR induction dataset of more than **11,000 compounds** using a low-cost, high-fidelity in-house assay. Compounds were sourced primarily from two Enamine libraries (Discovery Diversity 10 set and FDA Approved Drugs set) along with subsequent orders of follow-on compounds, and profiled through a rigorous multi-step assay flow reminiscent of an on-target drug discovery program.
The dataset was built through the following stages:
- **Primary Screen:** 11,362 diverse compounds screened at a single concentration.
- **Dose-Response:** 4,325 compounds selected for an 8-concentration dose-response (with extensive counter-screening in a PXR-null cell line to evaluate specificity).
- **Refinement:** 211 compounds showed EC50 ≤ 1 µM (pEC50 ≥ 6).
- **Counter-Screen:** 63 compounds selected based on minimal activity in a PXR-null cell line to confirm on-target specificity.
- **Analog Expansion Set:** Similarity searches (ECFP4 Tanimoto > 0.4) of these 63 actives yielded the **513-compound test set**, ordered from the Enamine US on-demand catalog and fully assayed with dose-response curves.
This design mimics a **lead optimization scenario**, shifting from broad hit-finding to detailed exploration of Structure-Activity Relationships (SAR). The analog set contains detailed SAR and activity cliffs that should prove challenging for models. Cumulatively, this represents the **largest PXR activity dataset available in the literature**.
### Developing the Assay
The assay employs a cell-based reporter system to measure PXR induction using a well-established two-part chimeric design. This fusion protein is composed of the ligand-binding domain (LBD) of human PXR attached to a heterologous DNA-binding domain, which acts via a reporter construct containing the corresponding DNA response element upstream of a luciferase gene. This approach reduces crosstalk, provides superior specificity, and maximizes signal-to-noise ratio compared to using the native PXR promoter. The system is stably integrated into the selected cell line to ensure consistent reporter activity across screening runs.
When a test compound acts as an agonist, it binds the PXR-LBD and induces a conformational change that promotes recruitment of transcriptional activators, driving luciferase expression. Multiple cell lines and genetic constructs were evaluated and optimized against reference compounds to achieve robust, reproducible data consistent with literature pEC50 values.
A parallel **counter-screen** using an identical reporter system with nonsense mutations in the chimeric PXR gene eliminates false positives — filtering out general transcriptional activators and HDAC inhibitors from true PXR agonists.
### How the Dataset Was Constructed
To determine the optimal concentrations for the primary screen, a pilot screen was run at 10, 30, and 100 µM. The 10 and 30 µM concentrations were selected, as their respective hit rates (~17% and ~51%) yielded a manageable set for follow-up dose-response curves while biasing toward the most potent compounds and mitigating solubility issues. A few thousand compounds yielded enough activity to be promoted to a full 8-point concentration dose response curve (DRC). By fitting these DRCs, the EC50s for these few thousand compounds were estimated. These were combined with data from initial direct-to-DRC experiments conducted early in the program's development, yielding a training dataset of 4,140 EC50 values.
For hit-calling, a linear model was fitted on logged data with a fixed effect term for each compound contrasted against the negative control. This assigns standard errors, p-values, and confidence intervals based on replicated control conditions. The Benjamini-Hochberg method was applied to control the false discovery rate (FDR). A compound is classified as a hit when its log₂ fold-change exceeds 1 and FDR < 5%.
Hit expansion targeted compounds with EC50 ≤ 1 µM and at least 1.5 log-unit difference between the primary assay pEC50 and the counterassay pEC50. This yielded 63 selective hits, from which analogs were selected from the Enamine US on-demand catalog with ECFP4 Tanimoto similarity > 0.4, forming the 513-compound test set.
```

## Setup

This setup is configured to use an LLM via `opencode`.

`opencode` supports arbitrary LLM's, but I am running `Qwen 3.6 27B` locally via `llama.cpp`.

For code dependencies, you will need `chemprop`, `molpipeline`, `xgboost`, and `scikit-learn`.

## Baseline

First, we establish a baseline model.
This was the prompt given to Qwen:

```
Build a machine learning model in `experiment.py` to predict the activity (pEC50) of various ligands against PXR.

In the @examples directory you have access to Jupyter notebook examples for training machine learning models using `chemprop` and `MolPipeline`. You also have access to `scikit-learn` and its usual collection of machine learning models.

The data is stored in `train.csv`. The data is sparse, and there are many other endpoints which were measured that are included in the training data, which can be used as pre-training or co-training targets or data filtering criterion, etc.

This is a one-shot setup - you should complete the code in `experiment.py` in whatever way you think will deliver the best machine learning model. The choice of data preprocessing, data selection, use of other endpoints, model architectures, model hyperparameters, ensembling, and all other considerations are up to you. You should read the Challenge Description in @README.md to understand more about the data, as well as using your own knowledge of machine learning.
```

## Model Development

From there, Qwen was prompted various times to use the [Autoresearch](./.opencode/commands/autoresearch.md) skill.
All experiments were logged with GitHub commits, which you can see [here](https://github.com/JacksonBurns/openadmet_pxr_qwen36/commits/main/).

## Final Model

The actual final model that was used is shown in [`experiment.py`](./experiment.py).
It is an ensemble of Chemprop, CheMeleon, and Ridge regression on the Osmordred descriptor set.
The LLM made some strange choices - it optimizes the weights of the ensemble during CV, but then does a naive average during evaluation.
This is probably because the latter _is better_ and it just didn't bother to remove the code.

To incorporate the phase 1 data, a residual model (see [`chemeleon_residual.sh`](./chemeleon_residual.sh)) was manually fit.
One could also just add this data to the training data and re-fit, but the human in the loop noticed that, no matter what was done, the models were terrible in the low pEC50 range.
The residual model is the most direct manner in which to address this, i.e., by directly showing the model where it is wrong.
At least, that's the hope.
