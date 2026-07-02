# Bibliography — Loss Space ≠ Representation Space

Derived from docs/report/ssl-projector-gap-report.html (compiled 2026-07-01, quote-verified there). Tags: [E] peer-reviewed & quote-verified · [P] preprint/abstract-level · [refuted] failed the report's adversarial verification — cited only to record exclusion.

**102 entries** (Method papers 18 · Projector/predictor role & the gap 16 · Representation metrics 9 · Failure modes & robustness 36 · Theory 23). Transcribed faithfully from the report's §8: author lists and titles are kept as the report abbreviates them; where the report gives no separate title, the bold short name is the title as given there. Two entries carry no arXiv ID in the report (α-ReQ; LeCun 2022, OpenReview only).

## Method papers

- **SimCLR** — Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) `[E]`
- **MoCo v1** — He, Fan, Wu, Xie, Girshick. *Momentum Contrast for Unsupervised Visual Representation Learning*. CVPR 2020. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722) `[E]`
- **MoCo v2** — Chen, Fan, Girshick, He. *Improved Baselines with Momentum Contrastive Learning*. tech report (2020, year from arXiv ID). [arXiv:2003.04297](https://arxiv.org/abs/2003.04297) `[E]`
- **MoCo v3** — Chen, Xie, He. *An Empirical Study of Training Self-Supervised Vision Transformers*. ICCV 2021. [arXiv:2104.02057](https://arxiv.org/abs/2104.02057) `[E]`
- **BYOL** — Grill et al. *Bootstrap Your Own Latent*. NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) `[E]`
- **SimSiam** — Chen & He. *Exploring Simple Siamese Representation Learning*. CVPR 2021. [arXiv:2011.10566](https://arxiv.org/abs/2011.10566) `[E]`
- **SwAV** — Caron et al. *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments*. NeurIPS 2020. [arXiv:2006.09882](https://arxiv.org/abs/2006.09882) `[E]`
- **DINO** — Caron et al. *Emerging Properties in Self-Supervised Vision Transformers*. ICCV 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) `[E]`
- **iBOT** — Zhou et al. *iBOT: Image BERT Pre-Training with Online Tokenizer*. ICLR 2022. [arXiv:2111.07832](https://arxiv.org/abs/2111.07832) `[E]`
- **DINOv2** — Oquab, Darcet, Moutakanni et al. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) `[E]`
- **Barlow Twins** — Zbontar, Jing, Misra, LeCun, Deny. *Self-Supervised Learning via Redundancy Reduction*. ICML 2021. [arXiv:2103.03230](https://arxiv.org/abs/2103.03230) `[E]`
- **VICReg** — Bardes, Ponce, LeCun. *Variance-Invariance-Covariance Regularization*. ICLR 2022. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) `[E]`
- **W-MSE** — Ermolov, Siarohin, Sangineto, Sebe. *Whitening for Self-Supervised Representation Learning*. ICML 2021. [arXiv:2007.06346](https://arxiv.org/abs/2007.06346) `[E]`
- **MAE** — He, Chen, Xie, Li, Dollár, Girshick. *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) `[E]`
- **I-JEPA** — Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) `[E]`
- **V-JEPA** — Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video*. TMLR 2024. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) `[E]`
- **LeJEPA** — Balestriero & LeCun. *LeJEPA: Provable and Scalable SSL Without the Heuristics*. preprint 2025. [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) `[P]`
- **SSL Cookbook** — Balestriero et al. *A Cookbook of Self-Supervised Learning*. 2023. [arXiv:2304.12210](https://arxiv.org/abs/2304.12210) `[E]`

## Projector / predictor role & the gap

- **Guillotine Regularization** — Bordes, Balestriero, Garrido, Bardes, Vincent. *Why removing layers is needed to improve generalization in SSL*. TMLR 2023. [arXiv:2206.13378](https://arxiv.org/abs/2206.13378) `[E]` — the central gap paper; >30pp; verified 3-0
- **RCDM** — Bordes, Balestriero, Vincent. *High Fidelity Visualization of What Your SSL Representation Knows About*. TMLR 2022. [arXiv:2112.09164](https://arxiv.org/abs/2112.09164) `[E]` — backbone not invariant; projector output is
- **Dimensional collapse / DirectCLR** — Jing, Vincent, LeCun, Tian. *Understanding Dimensional Collapse in Contrastive SSL*. ICLR 2022. [arXiv:2110.09348](https://arxiv.org/abs/2110.09348) `[E]` — 51.5/61.1/66.5/62.7 numbers verified 3-0
- **Rank Differential Mechanism** — Zhuo, Wang, Ma, Wang. *Unified Understanding of Non-contrastive Learning via RDM*. ICLR 2023. [arXiv:2303.02387](https://arxiv.org/abs/2303.02387) `[E]`
- **DirectPred** — Tian, Chen, Ganguli. *Understanding SSL Dynamics without Contrastive Pairs*. ICML 2021. [arXiv:2102.06810](https://arxiv.org/abs/2102.06810) `[E]`
- **Prediction-head mechanism** — Wen & Li. *The Mechanism of Prediction Head in Non-contrastive SSL*. NeurIPS 2022. [arXiv:2205.06226](https://arxiv.org/abs/2205.06226) `[E]` — substitution/acceleration; encoder-level guarantee; verified 3-0
- **Guillotine for supervised (t-ReX)** — Sariyildiz et al. *No Reason for No Supervision*. ICLR 2023 spotlight. [arXiv:2206.15369](https://arxiv.org/abs/2206.15369) `[E]`
- **Projection Head is Secretly an Information Bottleneck** — Ouyang, Hu, Zhang, Wang, Wang. ICLR 2025. [arXiv:2503.00507](https://arxiv.org/abs/2503.00507) `[E]` — pre-projector downstream guarantee
- **Investigating the Benefits of Projection Head** — Xue, Gan, Ni, Joshi, Mirzasoleiman. ICLR 2024. [arXiv:2403.11391](https://arxiv.org/abs/2403.11391) `[E]` — progressive feature weighting; verified 3-0
- **Deciphering the Projection Head (RED)** — Ma, Hu, Wang. IJCAI 2024. [arXiv:2301.12189](https://arxiv.org/abs/2301.12189) `[E]` — projector = uniformity projector; entropy numbers verified 3-0
- **How Does SimSiam Avoid Collapse** — Zhang et al. ICLR 2022. [arXiv:2203.16262](https://arxiv.org/abs/2203.16262) `[E]`
- **BYOL works even without batch statistics** — Richemond et al. 2020. [arXiv:2010.10241](https://arxiv.org/abs/2010.10241) `[E]`
- **The Edge of Orthogonality** — Richemond et al. *What Makes BYOL Tick*. ICML 2023. [arXiv:2302.04817](https://arxiv.org/abs/2302.04817) `[E]`
- **On the Importance of Asymmetry for Siamese Representation Learning** — Wang et al. CVPR 2022. [arXiv:2204.00613](https://arxiv.org/abs/2204.00613) `[E]`
- **Small projectors + multiple views** — Agrawal et al. NeurIPS 2024. [arXiv:2312.10725](https://arxiv.org/abs/2312.10725) `[E]`
- **The Geometry of Projection Heads** — Chaudhry. preprint 2026. [arXiv:2605.17180](https://arxiv.org/abs/2605.17180) `[refuted]` — refuted 1-2 / 1-2 / 0-3 — excluded; recorded for transparency

## Representation metrics

- **RankMe** — Garrido, Balestriero, Najman, LeCun. *Assessing Downstream Performance by Rank*. ICML 2023. [arXiv:2210.02885](https://arxiv.org/abs/2210.02885) `[E]` — necessary-not-sufficient; verified 3-0
- **α-ReQ** — Agrawal, Mondal, Ghosh, Richards. *Representation Quality by Eigenspectrum Decay*. NeurIPS 2022. (no arXiv ID given in the report) `[E]`
- **Alignment & Uniformity** — Wang & Isola. *Understanding Contrastive Learning on the Hypersphere*. ICML 2020. [arXiv:2005.10242](https://arxiv.org/abs/2005.10242) `[E]` — characterizes normalized loss space; verified 3-0
- **Duality of contrastive / non-contrastive** — Garrido, Chen, Bardes, Najman, LeCun. ICLR 2023. [arXiv:2206.02574](https://arxiv.org/abs/2206.02574) `[E]`
- **IdEst (intrinsic dimension)** — Mordacq, Kalogeiton, Oudot. 2026. [arXiv:2606.03338](https://arxiv.org/abs/2606.03338) `[P]`
- **Risk Decomposition** — Dubois, Hashimoto, Liang. *Evaluating SSL via Risk Decomposition*. ICML 2023 oral. [arXiv:2302.03068](https://arxiv.org/abs/2302.03068) `[E]`
- **CKA** — Kornblith, Norouzi, Lee, Hinton. *Similarity of Neural Network Representations Revisited*. ICML 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414) `[E]` — reliability refuted 0-3 — treat as contested (cf. Davari et al.)
- **AugSelf** — Lee et al. *Improving Transferability via augmentation-aware self-supervision*. NeurIPS 2021. [arXiv:2111.09613](https://arxiv.org/abs/2111.09613) `[E]`
- **CASSLE** — Przewięźlikowski et al. *Augmentation-aware SSL with Conditioned Projector*. KBS 2024. [arXiv:2306.06082](https://arxiv.org/abs/2306.06082) `[E]`

## Failure modes & robustness

- **Demystifying Contrastive SSL** — Purushwalkam & Gupta. *Invariances, Augmentations, Dataset Biases*. NeurIPS 2020. [arXiv:2007.13916](https://arxiv.org/abs/2007.13916) `[E]` — verified 2-1
- **What Should Not Be Contrastive (LooC)** — Xiao, Wang, Efros, Darrell. ICLR 2021. [arXiv:2008.05659](https://arxiv.org/abs/2008.05659) `[E]`
- **What Makes for Good Views (InfoMin)** — Tian et al. NeurIPS 2020. [arXiv:2005.10243](https://arxiv.org/abs/2005.10243) `[E]`
- **Object-Aware Cropping** — Mishra et al. TMLR 2022. [arXiv:2112.00319](https://arxiv.org/abs/2112.00319) `[E]` — +8.8 mAP object crops
- **Object-aware Contrastive (ContraCAM)** — Mo et al. NeurIPS 2021. [arXiv:2108.00049](https://arxiv.org/abs/2108.00049) `[E]`
- **Revisiting Contrastive Methods (scene-centric)** — Van Gansbeke et al. NeurIPS 2021. [arXiv:2106.05967](https://arxiv.org/abs/2106.05967) `[E]` — counterpoint
- **Multi-task latent-space objective (multi-crop conflict)** — De Plaen et al. 2026. [arXiv:2602.05845](https://arxiv.org/abs/2602.05845) `[P]`
- **Intriguing Properties of Contrastive Losses** — Chen, Luo, Li. NeurIPS 2021. [arXiv:2011.02803](https://arxiv.org/abs/2011.02803) `[E]` — feature suppression
- **Can contrastive learning avoid shortcut solutions?** — Robinson et al. NeurIPS 2021. [arXiv:2106.11230](https://arxiv.org/abs/2106.11230) `[E]`
- **Which Features are Learnt by Contrastive Learning?** — Xue et al. ICML 2023. [arXiv:2305.16536](https://arxiv.org/abs/2305.16536) `[E]`
- **Texture bias** — Geirhos et al. *ImageNet-trained CNNs are biased towards texture*. ICLR 2019. [arXiv:1811.12231](https://arxiv.org/abs/1811.12231) `[E]`
- **Shortcut Learning** — Geirhos et al. Nature Mach. Intell. 2020. [arXiv:2004.07780](https://arxiv.org/abs/2004.07780) `[E]`
- **Surprising similarities (SSL vs supervised)** — Geirhos et al. 2020. [arXiv:2010.08377](https://arxiv.org/abs/2010.08377) `[E]`
- **Partial success closing the human-machine gap** — Geirhos et al. NeurIPS 2021. [arXiv:2106.07411](https://arxiv.org/abs/2106.07411) `[E]`
- **Intriguing Properties of ViT** — Naseer et al. NeurIPS 2021. [arXiv:2105.10497](https://arxiv.org/abs/2105.10497) `[E]`
- **Backgrounds Challenge (IN-9)** — Xiao et al. *Noise or Signal*. ICLR 2021. [arXiv:2006.09994](https://arxiv.org/abs/2006.09994) `[E]`
- **Background Augmentations** — Ryali et al. 2021. [arXiv:2103.12719](https://arxiv.org/abs/2103.12719) `[E]`
- **Déjà Vu memorization** — Meehan et al. NeurIPS 2023. [arXiv:2304.13850](https://arxiv.org/abs/2304.13850) `[E]`
- **LA-SSL (spurious correlation)** — Zhu et al. NeurIPS 2023 wksp. [arXiv:2311.16361](https://arxiv.org/abs/2311.16361) `[E]`
- **Robustness of unsupervised learning to shift** — Shi et al. ICLR 2023. [arXiv:2206.08871](https://arxiv.org/abs/2206.08871) `[E]`
- **Whac-A-Mole** — Li et al. *Shortcuts Come in Multiples*. CVPR 2023. [arXiv:2212.04825](https://arxiv.org/abs/2212.04825) `[E]`
- **Hidden Uniform Cluster Prior** — Assran et al. ICLR 2023. [arXiv:2210.07277](https://arxiv.org/abs/2210.07277) `[E]`
- **SSL more robust to imbalance** — Liu et al. ICLR 2022. [arXiv:2110.05025](https://arxiv.org/abs/2110.05025) `[E]`
- **When Does Contrastive Learning Work?** — Cole et al. CVPR 2022. [arXiv:2105.05837](https://arxiv.org/abs/2105.05837) `[E]`
- **Divide and Contrast (uncurated)** — Tian et al. ICCV 2021. [arXiv:2105.08054](https://arxiv.org/abs/2105.08054) `[E]`
- **What Do SSL ViTs Learn? (CL vs MIM)** — Park et al. ICLR 2023. [arXiv:2305.00729](https://arxiv.org/abs/2305.00729) `[E]`
- **Dark Secrets of MIM** — Xie et al. CVPR 2023. [arXiv:2205.13543](https://arxiv.org/abs/2205.13543) `[E]`
- **Learning by Reconstruction is Uninformative** — Balestriero & LeCun. ICML 2024. [arXiv:2402.11337](https://arxiv.org/abs/2402.11337) `[E]`
- **Stochastic Positional Embeddings (StoP)** — Bar et al. ICML 2024. [arXiv:2308.00566](https://arxiv.org/abs/2308.00566) `[E]`
- **MIM-Refiner** — Alkin et al. ICLR 2025. [arXiv:2402.10093](https://arxiv.org/abs/2402.10093) `[E]`
- **C-JEPA** — Mo & Tong. *Connecting JEPA with Contrastive SSL*. NeurIPS 2024. [arXiv:2410.19560](https://arxiv.org/abs/2410.19560) `[E]`
- **Enhancing JEPAs with Spatial Conditioning** — Littwin, Thilak, Gopalakrishnan. NeurIPS 2024 wksp. [arXiv:2410.10773](https://arxiv.org/abs/2410.10773) `[E]`
- **DMT-JEPA** — Mo & Yun. 2024. [arXiv:2405.17995](https://arxiv.org/abs/2405.17995) `[E]`
- **How Well Do SSL Models Transfer?** — Ericsson et al. CVPR 2021. [arXiv:2011.13377](https://arxiv.org/abs/2011.13377) `[E]`
- **Contrasting Contrastive Pipelines** — Kotar et al. ICCV 2021. [arXiv:2103.14005](https://arxiv.org/abs/2103.14005) `[E]`
- **Contrastive SSL → higher adversarial susceptibility** — Gupta et al. AAAI 2023. [arXiv:2207.10862](https://arxiv.org/abs/2207.10862) `[E]`

## Theory

- **Contrastive learning guarantees** — Arora, Khandeparkar, Khodak, Plevrakis, Saunshi. ICML 2019. [arXiv:1902.09229](https://arxiv.org/abs/1902.09229) `[E]`
- **Understanding CL Requires Inductive Biases** — Saunshi et al. ICML 2022. [arXiv:2202.14037](https://arxiv.org/abs/2202.14037) `[E]` — loss-value analyses can be vacuous
- **Spectral Contrastive Loss** — HaoChen, Wei, Gaidon, Ma. NeurIPS 2021. [arXiv:2106.04156](https://arxiv.org/abs/2106.04156) `[E]`
- **Beyond Separability (linear transferability)** — HaoChen, Wei, Kumar, Ma. NeurIPS 2022. [arXiv:2204.02683](https://arxiv.org/abs/2204.02683) `[E]`
- **Inductive Biases in Contrastive Learning** — HaoChen & Ma. ICLR 2023. [arXiv:2211.14699](https://arxiv.org/abs/2211.14699) `[E]`
- **SSL recovers spectral embedding methods** — Balestriero & LeCun. NeurIPS 2022. [arXiv:2205.11508](https://arxiv.org/abs/2205.11508) `[E]`
- **Implicit variance regularization** — Halvagal, Laborieux, Zenke. NeurIPS 2023. [arXiv:2212.04858](https://arxiv.org/abs/2212.04858) `[E]`
- **Multi-View Information Bottleneck** — Federici et al. ICLR 2020. [arXiv:2002.07017](https://arxiv.org/abs/2002.07017) `[E]`
- **SSL from a Multi-view Perspective** — Tsai et al. ICLR 2021. [arXiv:2006.05576](https://arxiv.org/abs/2006.05576) `[E]`
- **Contrastive learning, multi-view redundancy, linear models** — Tosh, Krishnamurthy, Hsu. ALT 2021. [arXiv:2008.10150](https://arxiv.org/abs/2008.10150) `[E]`
- **Information-Theory Perspective on VICReg** — Shwartz-Ziv et al. NeurIPS 2023. [arXiv:2303.00633](https://arxiv.org/abs/2303.00633) `[E]`
- **Rethinking Minimal Sufficient Representation** — Wang, Guo, Deng, Lu. CVPR 2022 oral. [arXiv:2203.07004](https://arxiv.org/abs/2203.07004) `[E]`
- **The SSL Interplay** — Cabannes, Kiani, Balestriero, LeCun, Bietti. ICML 2023. [arXiv:2302.02774](https://arxiv.org/abs/2302.02774) `[E]`
- **Stepwise Nature of SSL** — Simon et al. ICML 2023. [arXiv:2303.15438](https://arxiv.org/abs/2303.15438) `[E]`
- **Barlow Twins ≈ HSIC** — Tsai, Bai, Morency, Salakhutdinov. 2021. [arXiv:2104.13712](https://arxiv.org/abs/2104.13712) `[E]`
- **How Mask Matters (U-MAE)** — Zhang, Wang, Wang. NeurIPS 2022. [arXiv:2210.08344](https://arxiv.org/abs/2210.08344) `[E]` — encoder-level guarantee for MAE
- **MAE via Hierarchical Latent Variable Models** — Kong et al. CVPR 2023 highlight. [arXiv:2306.04898](https://arxiv.org/abs/2306.04898) `[E]`
- **How to Understand Masked Autoencoders** — Cao, Xu, Clifton. 2022. [arXiv:2202.03670](https://arxiv.org/abs/2202.03670) `[E]`
- **A Path Towards Autonomous Machine Intelligence** — LeCun. 2022. [OpenReview BZ5a1r-kVsf](https://openreview.net/forum?id=BZ5a1r-kVsf) `[P]`
- **JEPA Focus on Slow Features** — Sobal et al. NeurIPS 2022 wksp. [arXiv:2211.10831](https://arxiv.org/abs/2211.10831) `[E]`
- **How JEPA Avoids Noisy Features** — Littwin et al. NeurIPS 2024. [arXiv:2407.03475](https://arxiv.org/abs/2407.03475) `[E]`
- **Understanding SSL via Latent Distribution Matching** — Mikulasch & Zenke. ICML 2026 Spotlight. [arXiv:2605.03517](https://arxiv.org/abs/2605.03517) `[E]` — existence & venue verified; unifies families as alignment+entropy
- **An Augmentation-Aware Theory for Contrastive SSL** — Cui, Wen, Wang. ICML 2025. [arXiv:2505.22196](https://arxiv.org/abs/2505.22196) `[E]`

---

BibTeX for all entries: `docs/literature/refs.bib`.
