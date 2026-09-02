# Citations & Data Usage — BraTS 2021

Reference this when writing the paper's references section and when setting up data access. Get data access and citation habits right from Day 1 — retrofitting citations later is error-prone.

---

## Data Access (Task 1 — Segmentation, what you need)

- **Registration:** create an account on the **Synapse platform** (synapse.org), then fill out the associated Google form linked from the official BraTS 2021 page to get access to the training data. Synapse is the official platform for Task 1 (segmentation) — do NOT use the Kaggle competition page, that's Task 2 (MGMT classification) and doesn't include segmentation-task imaging/masks in the format you need.
- **What you get:** training data includes ground truth segmentation annotations. Validation data (released separately) does *not* include annotations — you won't need it unless you want to benchmark against the official leaderboard later. Testing data is never released publicly.
- **Official page:** BraTS 2021 challenge page (CBICA, University of Pennsylvania) — go here for the current registration link and any updates.
- **Contact for data issues:** brats@cbica.upenn.edu

## Data Format

- NIfTI files (`.nii.gz`) for Task 1 (Segmentation) — this is what you want.
- DICOM files (`.dcm`) are for Task 2 (Classification) only — not relevant to your project.
- 4 modalities per subject: T1 (native), T1Gd (post-contrast T1-weighted, i.e. T1ce), T2, T2-FLAIR.
- All scans co-registered to the same anatomical template, resampled to 1mm³ isotropic resolution, and skull-stripped — confirms what was assumed in the preprocessing plan (Week 1, Day 2): skull-stripping is already done, don't redo it.
- Segmentation labels: **GD-enhancing tumor (ET) = label 4**, **peritumoral edema (ED) = label 2**, **necrotic tumor core (NCR) = label 1**, background = 0. (Matches the label mapping already noted in the Week 1 checklist, Day 1 — confirmed against the primary source now.)

## Required Citations

**Every use of BraTS data — in the paper, in any writeup, anywhere — must cite all three of these:**

1. U. Baid, S. Ghodasara, S. Mohan, M. Bilello, E. Calabrese, E. Colak, K. Farahani, J. Kalpathy-Cramer, F. C. Kitamura, S. Pati, et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification," *arXiv:2107.02314*, 2021.

2. B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)," *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024, 2015. DOI: 10.1109/TMI.2014.2377694

3. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features," *Nature Scientific Data*, 4:170117, 2017. DOI: 10.1038/sdata.2017.117

**If your target journal/conference has no restriction on citing "data citations" separately, also add these two** (recommended to include regardless, for full reproducibility credit to data contributors):

4. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection," *The Cancer Imaging Archive*, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

5. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection," *The Cancer Imaging Archive*, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF

## Data Usage Terms — What You're Allowed / Not Allowed to Do

- You may use and refer to BraTS data freely in your own research, **provided you always cite the three (or five) references above.**
- You may use **additional public data** to extend BraTS for training. You may **not** use additional *private* data (from your own institution) for the model you intend to have ranked/compared, and you may **not** use models pretrained on private datasets — this is a fairness constraint from the challenge, not strictly binding on your independent research project, but worth respecting anyway since it's the norm the field compares against.
- If you do use extra public/private data for your paper specifically (not for a ranked challenge submission), you must explicitly state this in your manuscript and separately report results using BraTS'21 data alone, so readers can see the isolated effect.
- Using BraTS results for MLPerf.org benchmark submissions is considered non-commercial use — not relevant to you unless you go that route later.

## Prior-Work Citations (from earlier planning — collect BibTeX as you read)

These are the papers from the reading itinerary (`project_plan.md`, Section 2 & 4) you'll also need to cite in your related-work section. Add exact BibTeX entries here as you read each one, so citation-collection doesn't become a last-week scramble:

- [ ] HeMIS (Havaei et al., 2016)
- [ ] U-HVED (Dorent et al., 2019)
- [ ] RFNet (Ding et al., 2021)
- [ ] ACN (Wang et al., 2021)
- [ ] mmFormer (Zhang et al., MICCAI 2022)
- [ ] Attention Is All You Need (Vaswani et al., 2017)
- [ ] 3D U-Net (Çiçek et al., 2016)
- [ ] nnU-Net (Isensee et al., 2021)
- [ ] ViT (Dosovitskiy et al., 2020)
- [ ] UNETR (Hatamizadeh et al., 2022)

---

## Quick Reference: Don't Confuse These

| | BraTS 2021 Task 1 (what you want) | RSNA-MICCAI Kaggle (Task 2) |
|---|---|---|
| Access via | Synapse | Kaggle |
| Format | NIfTI | DICOM |
| Labels | Segmentation masks (WT/TC/ET) | MGMT methylation status (binary) |
| Use for this project | Yes — this is your dataset | No |
