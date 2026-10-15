# TRIPOD+AI checklist

**Article:** CT-HEG: timestamp-conditioned message passing for post-48-hour ICU mortality prediction — an architectural ablation study

**Authors:** Mohammad Nasir Uddin, Rahnuma Tabassum Orpita, Asaduzzaman Anik, Eklachur Rahman Bhuiyan, Marjahan Risalat, S. M. Wali Ullah, Asif Ahamed

**Study type:** Model development with internal (held-out split) evaluation (D;E)

**Checklist source:** Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. Item wording below is abbreviated; refer to the original statement for the full text.

**Location** refers to the section headings of the manuscript. **Status:** Reported; Partial (some elements reported); Not reported; N/A (not applicable to this study).

| Section | Item | Checklist item (abbreviated) | Status | Location / note |
|:--|:--|:--|:--|:--|
| Title | 1 | Identify the study as developing or evaluating a prediction model, the target population, and the outcome | Reported | Title |
| Abstract | 2 | See TRIPOD+AI for Abstracts | Partial | Abstract (structured to the F1000Research format; not all abstract-checklist elements are included) |
| Introduction: background | 3a | Healthcare context and rationale, including existing models | Reported | Introduction; Related Work; Table 1 |
| | 3b | Target population, intended purpose in the care pathway, and intended users | Partial | Introduction; Task Definition. Intended users are not specified because the study is an architectural ablation, not a model proposed for deployment |
| | 3c | Known health inequalities between sociodemographic groups | Not reported | — |
| Objectives | 4 | Study objectives, including whether development, validation, or both | Reported | Introduction (contributions) |
| Methods: data | 5a | Data sources, rationale, and representativeness | Reported | Data source, ethics and study registration; Datasets and Cohort Definitions; Limitations (Calibration and distribution shift) |
| | 5b | Dates of data collection and end of follow-up | Partial | MIMIC-IV v3.1 and the 48-hour observation window are stated; calendar dates of the admissions are not stated |
| Methods: participants | 6a | Study setting, number and location of centers | Reported | Data source (single center, Beth Israel Deaconess Medical Center, Boston, MA, USA) |
| | 6b | Eligibility criteria | Reported | Datasets and Cohort Definitions (adults, first ICU stay per patient, length of stay ≥ 48 h) |
| | 6c | Treatments received and how they were handled | Partial | Medication administrations are model inputs (The Continuous-Time Heterogeneous EHR Graph); other treatments not handled explicitly |
| Data preparation | 7 | Pre-processing and quality checks; whether similar across groups | Partial | Offline Graph Construction; Type-Specific Input Embeddings; Time-Attentive Edge Encoding (log transform, z-scoring, clipping); Limitations (placeholder codes not removed). Consistency across sociodemographic groups not assessed |
| Methods: outcome | 8a | Outcome definition, time horizon, how and when assessed, rationale | Reported | Task Definition (post-48-hour in-hospital mortality). Consistency across groups not assessed |
| | 8b | Qualifications of outcome assessors (if subjective) | N/A | Outcome is recorded in-hospital mortality |
| | 8c | Blinding of outcome assessment | N/A | Outcome is recorded in-hospital mortality |
| Methods: predictors | 9a | How initial predictors were chosen; any pre-selection | Partial | Modalities and the exclusion of diagnosis codes are justified (Leakage Audit: Diagnosis-Code Timing); selection of the individual vital-sign channels and laboratory analytes is not described in the manuscript |
| | 9b | Definition of all predictors, including how and when measured | Partial | Node and edge definitions and the 48-hour window are stated (Problem Formulation and the CT-HEG Schema); the full variable list is in the released code, not the manuscript |
| | 9c | Qualifications of predictor assessors (if subjective) | N/A | Predictors are recorded EHR measurements |
| Sample size | 10 | How study size was arrived at and its adequacy | Not reported | No sample-size calculation; all eligible stays were used |
| Missing data | 11 | Handling of missing data and reasons for omission | Reported | Offline Graph Construction (no imputation; an unobserved channel contributes no node or edge) |
| Methods: analytical methods | 12a | How data were used and partitioned for development and evaluation | Reported | Splits and Training Protocol |
| | 12b | Handling of predictors (functional form, rescaling, transformation) | Reported | Type-Specific Input Embeddings; Time-Attentive Edge Encoding |
| | 12c | Model type, rationale, building steps, hyperparameter tuning, internal validation | Reported | CHIRP-Net sections; Splits and Training Protocol; Table 2 (hidden size selected on validation from 80 and 192) |
| | 12d | Heterogeneity across clusters | N/A | Single center |
| | 12e | Performance measures and plots, with rationale; comparison of models | Reported | Predictive Metrics and Statistical Testing; Participant Characteristics, Subgroup and Calibration Analyses |
| | 12f | Model updating (e.g., recalibration) after evaluation | N/A | No updating or recalibration performed |
| | 12g | How predictions were calculated for evaluation | Reported | Software availability (training and evaluation code) |
| Class imbalance | 13 | Class-imbalance methods, rationale, and recalibration | Partial | Splits and Training Protocol (class-balanced WeightedRandomSampler); Table 2 (class-weighted logistic regression); its effect on calibration is reported (Calibration; Table 9), but no recalibration was applied |
| Fairness | 14 | Approaches to address model fairness | Partial | Participant Characteristics, Subgroup and Calibration Analyses; Table 8 (subgroup discrimination by sex, age, race/ethnicity and insurance). Subgroup calibration and threshold-based fairness metrics not assessed (Limitations) |
| Model output | 15 | Form of model output and rationale for any thresholds | Reported | Readout and Prediction (predicted probability); no classification threshold used |
| Training vs. evaluation | 16 | Differences between development and evaluation data | Reported | Splits and Training Protocol (stratified random split from the same source) |
| Ethical approval | 17 | Ethics committee and consent or waiver | Reported | Data source, ethics and study registration; Ethics and consent |
| Open science | 18a | Funding source and role | Reported | Grant information |
| | 18b | Conflicts of interest | Reported | Competing interests |
| | 18c | Protocol | Reported | Data source, ethics and study registration (no protocol published) |
| | 18d | Registration | Reported | Data source, ethics and study registration (not preregistered) |
| | 18e | Data sharing | Reported | Data availability |
| | 18f | Code sharing | Reported | Software availability |
| Patient and public involvement | 19 | Patient and public involvement | Reported | Data source, ethics and study registration (none) |
| Results: participants | 20a | Participant flow, numbers with and without the outcome | Partial | Datasets and Cohort Definitions; Splits and Training Protocol (deaths/survivors per split); no flow diagram with exclusion counts |
| | 20b | Participant characteristics, including predictors, outcomes, missing data, and demographic differences | Reported | Participant Characteristics; Table 7 (overall, survivors, non-survivors, test set) |
| | 20c | Comparison of evaluation with development data distributions | Reported | Table 7 (test set compared with the full cohort); mortality per split (Splits and Training Protocol) |
| Results: model development | 21 | Number of participants and outcome events in each analysis | Reported | Splits and Training Protocol (stays and deaths in training, validation and test sets) |
| Results: model specification | 22 | Full model details to enable predictions and reuse | Partial | Architecture and protocol (Methods; Table 2); code (Software availability); checkpoints available on request to credentialed researchers |
| Results: model performance | 23a | Performance estimates with confidence intervals, including key subgroups | Reported | Tables 4–6 and 8–9; Figure 2 (reliability diagram); subgroup AUROC/AUPRC with bootstrap CIs; baseline comparisons are descriptive without CIs |
| | 23b | Heterogeneity across clusters | N/A | Single center |
| Results: model updating | 24 | Results of model updating | N/A | No updating performed |
| Discussion: interpretation | 25 | Overall interpretation, including fairness, in light of objectives and previous studies | Reported | Discussion; Limitations (Subgroup performance and fairness) |
| Discussion: limitations | 26 | Limitations and their effect on bias, uncertainty, and generalizability | Reported | Limitations |
| Discussion: usability | 27a | Handling of poor-quality or unavailable input data in implementation | Not reported | Model not proposed for implementation |
| | 27b | Whether users interact with input data and required expertise | Not reported | Model not proposed for implementation |
| | 27c | Next steps for research, applicability, and generalizability | Reported | Limitations (Calibration and distribution shift); Generalizability Beyond ICU Mortality |
