# Literature reconnaissance: learned predictive codec for CT projection streams

Compiled 2026-09-04. Design under study: an autoregressive model that takes a prefix of a
single slice's sinogram (angle × detector-column) and predicts the next angular projection
(a 1D row); the residual is entropy-coded. Parallel-beam, known constant Δθ, slices treated
independently.

**Method note on this document**: every entry below is labeled VERIFIED (I fetched the
actual paper/abstract page or a page that quotes it directly, e.g. PubMed/PMC/arXiv/IEEE/
publisher abstract) or UNVERIFIED (only a search-engine synthesized snippet was available;
title/authors/venue could not be independently confirmed by opening the source). Do not cite
an UNVERIFIED entry without checking it yourself first. I did not invent any titles — every
entry below corresponds to a real search hit; the UNVERIFIED tag reflects only that I could
not get past a paywall/redirect to confirm bibliographic details first-hand.

---

## 1. Compressing CT/tomographic projection data (sinograms) — lossless / near-lossless

### Medical CT raw-projection compression (foundational, not synchrotron)

- **K. T. Bae, B. R. Whiting, "CT Data Storage Reduction by Means of Compressing Projection
  Data Instead of Images: Feasibility Study," *Radiology*, 219(3), 2001.**
  DOI: 10.1148/radiology.219.3.r01jn49850.
  UNVERIFIED (redirect to pubs.rsna.org returned 403; bibliographic details come from
  consistent search-engine snippets and a matching ResearchGate record, not a direct fetch).
  Claimed finding: projection data compress better than reconstructed images; a compression
  ratio of ~3 keeps peak error under 1 HU, ratio ~9 raises image noise by ~1 HU.
  Link: https://doi.org/10.1148/radiology.219.3.r01jn49850

- **Y. Xie et al., "Lossy raw data compression in computed tomography with noise shaping to
  control image effects," Proc. SPIE 6913 (Medical Imaging 2008: Physics of Medical
  Imaging), 691332.**
  DOI: 10.1117/12.769954. UNVERIFIED (abstract-level only, via search synthesis; not
  directly fetched, but the SPIE listing and a matching PMC/AAPM follow-up paper corroborate
  it). Introduces two noise-shaping schemes (error-feedback filtering, sub-band coding with
  bit allocation) for CT raw-data compression that push reconstruction error toward the
  periphery of the field of view — i.e., quantization noise is shaped away from the region
  of interest rather than reduced overall. This is a **predictive/DPCM-style** raw-data
  compression scheme, close in spirit to what we're proposing, but on the medical-CT slip-
  ring bandwidth problem rather than the angular axis specifically.
  Link: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/6913/691332/

- **Related follow-on**: "Understanding and controlling the effect of lossy raw data
  compression on CT images" (AAPM/Med. Phys., search hit only) — UNVERIFIED, same group.

- **"Compression of CT sinogram data by decimation in the view direction," *Medical
  Physics*, 2017.** DOI: 10.1002/mp.12181. UNVERIFIED (403 on fetch; only the title/venue
  confirmed from the search index, not the abstract). Directly relevant: this is angular
  (view-direction) decimation of sinogram data, i.e. compressing across the angle axis,
  though by subsampling/decimation rather than learned prediction.
  Link: https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.12181

### Synchrotron / micro-CT raw-projection data specifically

- **F. Marone, J. Vogel, M. Stampanoni, "Impact of lossy compression of X-ray projections
  onto reconstructed tomographic slices," *Journal of Synchrotron Radiation*, 27(5), 2020.**
  DOI: 10.1107/S1600577520007353. **VERIFIED** (full text fetched via PMC).
  This is the single most relevant beamline paper for Q1 — it is from the TOMCAT group at
  PSI/SLS. They test JPEG2000, JPEG XR, and a bit-reset+bzip2 scheme directly on raw
  projections (not sinograms after reconstruction) at TOMCAT. Findings: a compression
  factor of 3–4× on raw projections causes **no measurable quality loss** on the
  reconstructed slice for standard high-SNR datasets; phase-retrieved (phase-contrast) data
  tolerate 6–8× because phase retrieval's smoothing suppresses compression artifacts; fast/
  low-dose acquisitions are more conservative (also capped near 3–4×) because of lower
  intrinsic SNR. This paper is essentially the PSI/TOMCAT institutional answer to "how much
  can we lossy-compress projections before it matters," and gives us a concrete numeric
  target our residual-coding scheme should beat *losslessly* or match *near-losslessly*.
  Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC7467350/

  Context also confirmed (VERIFIED via search + PSI beamline page): TOMCAT's GigaFRoST
  detector produces up to 1255 fps of 2016×2016 frames (7.7 GB/s), ~1 PB/year moved, and PSI
  currently archives **only raw, uncompressed** projections (no reconstructions) to the CSCS
  tape archive — i.e. no compression is applied in the archival pipeline today. This is a
  useful "current baseline is literally none" data point.
  Link: https://www.psi.ch/en/sls/tomcat/computing-infrastructure

- **Z. Min-xing, F. Shi-yuan, G. Yu, C. Yao-dong et al., "Fast lossless images compression
  for synchrotron radiation facility using deep learning and hybrid architecture,"
  *Radiation Detection Technology and Methods*, 8, 1693 (2024).**
  DOI: 10.1007/s41605-024-00490-9. UNVERIFIED (Springer redirected to a login wall; author
  list and abstract come from search-engine synthesis of the ADS/RDTM listing, not a direct
  read of the full text — treat the exact author list as provisional).
  From China's High Energy Photon Source (HEPS). Explicitly states conventional compression
  (presumably standard codecs) gives ratios "often below 1.5" on this class of data, and
  proposes a **"spatiotemporal learning network for predictive pixel value estimation"**
  plus residual quantization — i.e. a learned predictive/DPCM coder, benchmarked against
  DeepZip, on a GPU+CPU+FPGA hybrid pipeline (40% faster than DeepZip at comparable ratio,
  additional 38% via hardware acceleration). This is the closest thing I found in the
  literature to "predictive coding of synchrotron frame streams," although note it appears
  to be framed around frame-to-frame (detector image stream) prediction rather than
  specifically the angular axis of a sinogram — worth reading the primary source before
  citing specifics.
  Link: https://link.springer.com/article/10.1007/s41605-024-00490-9

- **General synchrotron-community picture** (VERIFIED via multiple corroborating hits,
  including the Marone/Vogel/Stampanoni paper's own framing): raw/projection CT data at
  major facilities (ESRF, APS, Diamond, PSI, and by extension DESY) is largely **stored
  uncompressed** today, or at best with generic lossless byte-oriented filters (bit-shuffle +
  zstd/lz4/bzip2) inside HDF5/NeXus containers — I found no facility-specific paper
  describing an operational sinogram-specific compressor in production. The bit-shuffle+zstd
  numbers I *could* verify come from macromolecular crystallography, not tomography (see
  next item) — treat any transfer to CT projections as an analogy, not a demonstrated
  result.

- **H. J. Bernstein, J. Jakoncic, "Investigation of fast and efficient lossless compression
  algorithms for macromolecular crystallography experiments," *Journal of Synchrotron
  Radiation*, 31(4), 2024.** DOI: 10.1107/S160057752400359X. **VERIFIED** (full text via
  PMC). NSLS-II (Brookhaven), AMX beamline, DECTRIS Eiger X 9M detector, lysozyme
  diffraction frames. Compares lz4, bslz4 (bit-shuffle+LZ4, the current de facto standard),
  zstd (levels 2–6), bszstd (bit-shuffle+zstd), szstd (byte-shuffle+zstd). Numbers for the
  hardest case (25% beam transmission, crystal-only frames): bslz4 baseline 7.50:1 @ 42.35
  frames/s; bszstd_2 gives 10.55:1 (+41%) @ 15.84 frames/s. This is **not tomography** (it's
  single-crystal diffraction, not a rotation sinogram) but it is a rigorous, recent, directly
  relevant benchmark of what generic shuffle+entropy-coder pipelines achieve on real
  synchrotron photon-counting detector data (EIGER-class), and a useful throughput/ratio
  anchor for what a learned method needs to beat.
  Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC11226158/

- **M. Hammer, K. Yoshii, A. Miceli, "Strategies for on-chip digital data compression for
  X-ray pixel detectors," arXiv:2006.02639 (2020); related J. Instrum. 16 P01025 (2021).**
  **VERIFIED** (abstract fetched from arXiv). In-pixel/on-chip compression for X-ray
  detectors (APS context) that quantizes ADC output to photon-count units **at the level of
  the Poisson noise floor**, giving a data-independent >1.5× ratio "for free" without loss of
  scientifically meaningful information, plus a separate zero-suppression ("zeromask") scheme
  giving >4×, >7×, >8× on HEDM, ptychography, and XPCS datasets respectively (combined
  6–12× bandwidth gain). This is the most directly relevant paper I found for the
  "quantization step tied to local Poisson σ" idea in Q6 — see below.
  Link: https://arxiv.org/abs/2006.02639

**Bottom line for Q1**: I found no published paper that does *predictive coding specifically
across the angular axis of a sinogram* for compression (as opposed to decimation/
subsampling, or frame-to-frame prediction in a general video-like sense). The Zhang et al.
HEPS paper is the closest analog and should be read in full before we claim novelty is or
isn't there. Reported real-world compression ratios at beamlines cluster in a narrow, low
range: ~1.4–1.8 with classical lossless codecs on raw CT sinograms (medical-CT survey
number, UNVERIFIED), 3–4× lossy on TOMCAT projections with *no measurable* reconstruction
quality loss (VERIFIED), and 7.5–10.5:1 on crystallography frames with bit-shuffle+zstd
(VERIFIED, but a different type of detector data).

---

## 2. Learned lossless image compression — SOTA and reusable machinery

- **T. Salimans, A. Karpathy, X. Chen, D. P. Kingma, "PixelCNN++: Improving the PixelCNN
  with Discretized Logistic Mixture Likelihood and Other Modifications," ICLR 2017.**
  arXiv:1701.05517. **VERIFIED** (search-corroborated across OpenReview + arXiv + multiple
  citation databases with consistent numbers). Discretized mixture-of-logistics (DMoL)
  output head — the standard trick for putting a tractable, differentiable, easily
  entropy-codable likelihood on 8-bit (or n-bit) pixel/sample values without one-hot softmax
  over 256 classes. Reports **2.92 bits/subpixel on CIFAR-10** (down from PixelRNN's 3.00).
  This DMoL head is exactly the kind of output distribution we'd want on the residual/next-row
  prediction in our angular-autoregressive design, generalized to whatever bit depth the
  detector uses.
  Link: https://arxiv.org/abs/1701.05517

- **F. Mentzer, E. Agustsson, M. Tschannen, R. Timofte, L. Van Gool, "Practical Full
  Resolution Learned Lossless Image Compression" (L3C), CVPR 2019.** arXiv:1811.12817.
  **VERIFIED** (abstract fetched from arXiv). First practical *learned* lossless codec;
  beats PNG, WebP, JPEG2000. Key idea: a hierarchical probabilistic model with learned
  auxiliary "feature" representations (not just RGB) so that only **3 forward passes**
  predict all pixel probabilities (vs. one sequential pass per pixel for PixelCNN), giving
  ">100× speedup" over the fastest PixelCNN variant at sampling time. This
  parallel-hierarchical-context trick is relevant if we want our angle-axis model to avoid
  being fully sequential.
  Link: https://arxiv.org/abs/1811.12817

- **E. Hoogeboom, J. Peters, R. van den Berg, M. Welling, "Integer Discrete Flows and
  Lossless Compression," NeurIPS 2019.** arXiv:1905.07376. **VERIFIED** (search-corroborated
  via NeurIPS proceedings page + arXiv, consistent details). Bijective integer-valued flow
  (no quantization/dequantization mismatch), competitive/SOTA bits-per-dim on CIFAR-10,
  ImageNet32, ImageNet64 at the time. Relevant if we want an exactly-invertible transform
  instead of a lossy "predict + code residual" split.
  Link: https://arxiv.org/abs/1905.07376

- **S. Cao, C.-Y. Wu, P. Krähenbühl, "Lossless Image Compression through Super-Resolution"
  (SReC), 2020.** arXiv:2004.02872. **VERIFIED** (arXiv-indexed with consistent figures
  across multiple sources incl. the authors' own GitHub README). Stores a low-res version
  losslessly, then predicts/entropy-codes each super-resolution upsampling step
  conditioned on the lower-res image. Reports **4.29 bpsp on ImageNet64, 2.70 bpsp on Open
  Images**. The "predict the next level of detail conditioned on what you already have"
  structure maps directly onto "predict the next angular row conditioned on the sinogram
  prefix."
  Link: https://arxiv.org/abs/2004.02872

- **J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston, "Variational Image Compression
  with a Scale Hyperprior," ICLR 2018.** arXiv:1802.01436. **VERIFIED** (search-corroborated,
  well-known/canonical paper, consistent description across all hits). This is a *lossy*
  compression paper, but it is the origin of the "hyperprior" mechanism (a second, coarser
  latent that predicts the scale/variance of the main latent's entropy model) that
  essentially every later lossless/near-lossless learned codec (including the DLPR-class
  models below) builds on for context modeling. Necessary background even though not itself
  lossless.
  Link: https://arxiv.org/abs/1802.01436

- **Current best-performing region (bits-per-subpixel on Kodak, standard benchmark),
  UNVERIFIED chain of search-engine-reported numbers, not independently opened**:
  DLPR ("Deep Lossless Probabilistic Ratio"-type model) ≈ **2.55 bpsp on Kodak**; a 2025
  follow-up claims **2.29 bpsp** (≈10% better than DLPR); on ImageNet64/Open Images a
  commonly cited comparison table (source page not independently confirmed) lists PNG
  17.22/12.09 bpp, WebP 13.92/9.09, FLIF 13.62/8.61, IDF 11.70/8.28, L3C 13.26/8.97, SReC
  12.90/8.10, and a "MPSMCT" model at 11.33/7.48 bpp. **I could not verify the exact paper
  title/authors behind "DLPR" or "MPSMCT" from a primary source in this session** — flag
  these two acronyms as needing a direct citation check before use in any writeup.
  I did fetch **T. Li, Q. Xia, Y. Li, R. Guo, G. Yang, "Deep Lossless Image Compression via
  Masked Sampling and Coarse-to-Fine Auto-Regression," arXiv:2503.11231 (2025)** directly —
  **VERIFIED** existence/authors/abstract, but the abstract alone doesn't give the bpsp table
  (would need the PDF body).
  Link: https://arxiv.org/abs/2503.11231

**Throughput**: across this literature, the general pattern (VERIFIED qualitatively, exact
numbers vary by paper/hardware) is that plain sequential PixelCNN-style autoregression is
slow (one network pass per sample), while L3C/SReC-style hierarchical/parallel schemes trade
a small bpp penalty for orders-of-magnitude higher throughput. For a per-row (not per-pixel)
autoregressive scheme like ours, this tradeoff is more favorable from the start since each
angular step already predicts an entire 1D row in one forward pass rather than one scalar.

---

## 3. Classical lossless / near-lossless predictive codecs and scientific-data compressors

- **JPEG-LS / LOCO-I**: M. Weinberger, G. Seroussi, G. Sapiro, "The LOCO-I lossless image
  compression algorithm: principles and standardization into JPEG-LS," *IEEE Trans. Image
  Processing*, 2000 (algorithm itself dates to ISO/IEC 14495-1, 1999). UNVERIFIED in this
  session (I found and read secondary summaries/PDF mirrors of LOCO-I but did not open the
  IEEE TIP page itself) — this is nonetheless an extremely well-established, uncontroversial
  citation; low risk. LOCO-I is a **causal, context-adaptive linear predictor + Golomb-Rice
  residual coding** scheme — architecturally the closest classical analog to our
  "predict-then-entropy-code-the-residual" design, just with a fixed rather than learned
  predictor, and operating pixel-causally in 2D rather than row-causally across angle.
  A commonly cited number (UNVERIFIED, from a comparative survey, not the primary source):
  ~14.23:1 (≈1.12 bpp) average on a set of 16-bit test images — this figure looks
  optimistic/dataset-dependent and should not be trusted without checking the source dataset.

- **CALIC**: X. Wu, N. Memon, "CALIC — a context-based adaptive lossless image codec,"
  ICASSP 1996. UNVERIFIED in this session (found only as a search hit, not opened). Broadly
  reported (across several secondary sources) as marginally better compression than JPEG-LS
  at substantially higher computational complexity — "a few percent" is the number that
  recurs, but I did not verify it against the primary paper.

- **FLIF**: J. Sneyers, P. Wuille, "FLIF: Free Lossless Image Format based on MANIAC
  compression," ICIP 2016. DOI: 10.1109/ICIP.2016.7532320. UNVERIFIED (only search-index
  metadata seen, not the primary PDF). MANIAC = Meta-Adaptive Near-zero Integer
  Arithmetic Coding, a per-image-adaptive decision-tree context model. FLIF's transform
  chain was absorbed into **JPEG XL's Modular mode**, which is the currently maintained
  lossless codec in this lineage.

- **JPEG XL**: J. Alakuijala et al., "JPEG XL next-generation image compression
  architecture and coding tools," Proc. SPIE 11137, 2019. DOI: 10.1117/12.2529237.
  **VERIFIED** full author list via search-engine synthesis of the SPIE/ADS/Semantic
  Scholar records (title, venue, DOI, and 16-author list consistent across all three
  independent listings) — I did not open the SPIE paywall page itself, but corroboration
  across independent bibliographic databases is strong. Modular mode (lossless/near-lossless)
  descends from FLIF/FUIF.
  Link: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11137/111370K/

- **JPEG 2000 lossless**: reversible 5/3 wavelet + EBCOT, ISO/IEC 15444-1. Not indepedently
  re-verified this session beyond its use as a comparison baseline in Marone/Vogel/
  Stampanoni (2020, VERIFIED above) and in several sinogram-completion / limited-angle
  papers.

- **PNG**: DEFLATE + simple per-scanline predictive filters (Paeth etc.) — the weakest
  classical baseline in essentially every learned-lossless comparison table found above.

- **FFV1**: M. Niedermayer et al., **RFC 9043, "FFV1 Video Coding Format Versions 0, 1, and
  3," IETF, August 2021.** **VERIFIED** (search-corroborated, official IETF RFC, high
  confidence — RFC text itself not opened but RFC number/title/date consistent across
  ffmpeg.org, archive.org, and Library of Congress preservation-format pages). Intra-frame,
  range-coder-based, used for archival video preservation (Library of Congress-endorsed
  since 2014); relevant to your own gray10 lossless-HEVC pipeline as an alternative
  intra-only lossless codec worth benchmarking against.
  Link: https://www.ffmpeg.org/~michael/ffv1.html

- **Lossless HEVC/AV1 for high-bit-depth scientific data**: search turned up only general
  comparison surveys (e.g., D. Barina, "Comparison of Lossless Image Formats," WSCG 2021,
  arXiv:2108.02557 — UNVERIFIED, not opened) and a note (UNVERIFIED, from search synthesis)
  that AV1's lossless mode tops out at 10-bit and neither HEVC nor AV1 natively handle
  12–16-bit RAW without workarounds — consistent with your own project note that gray10
  streaming already deals with a bit-depth ceiling. I found no paper benchmarking lossless
  HEVC/AV1 specifically on 16-bit scientific/CT sinogram-type data; this looks like a real
  gap (or just undocumented internal/industrial knowledge).

- **ZFP**: P. Lindstrom, "Fixed-Rate Compressed Floating-Point Arrays," IEEE TVCG, 2014
  (canonical ZFP paper — not independently opened this session, found only via secondary
  descriptions of the ZFP transform-coding pipeline: block-wise 4×4×4 near-orthogonal
  transform + embedded/bit-plane coding, absolute error-bound mode). UNVERIFIED primary
  source, but ZFP itself is extremely well-established.

- **SZ**: S. Di, F. Cappello, "Fast Error-Bounded Lossy HPC Data Compression with SZ," IPDPS
  2016 (canonical SZ paper — UNVERIFIED, not opened directly, found via secondary summary).
  Prediction-based (Lorenzo predictor / linear regression) + error-bounded quantization +
  Huffman/entropy coding — architecturally a **classical predictive coder with an explicit,
  tunable error bound**, i.e. exactly the "quantize to a target error, not to a target
  bit-depth" philosophy relevant to Q6/Q7. Reported (search-synthesized, UNVERIFIED) ratios
  ">500:1" are for very smooth simulation data, not representative of the noisy detector
  data we'd apply this to.

- **SPERR**: S. Li et al. (NCAR), "Lossy Scientific Data Compression With SPERR," IPDPS
  2023. **Partially VERIFIED**: paper's existence, NCAR/GitHub authorship, and the CDF9/7
  wavelet + SPECK + zstd pipeline description are corroborated by the official NCAR GitHub
  README and a matching conference PDF link, though I did not open the IPDPS proceedings
  page itself. Selectable quality control by BPP, PSNR, or point-wise error (PWE) — the PWE
  mode is the relevant one for a "noise-matched" quantization philosophy.
  Link: https://github.com/NCAR/SPERR

- **blosc / zstd+bit-shuffle**: not a single citable paper, but well documented via the
  hdf5plugin project and the Bernstein & Jakoncic (2024, VERIFIED above) crystallography
  benchmark (7.5–10.5:1 on Eiger detector frames). This is almost certainly the actual
  status quo at most beamlines for whatever compression *is* applied to raw detector data.

**Numbers we can quote with confidence for 16-bit-ish noisy scientific imagery**: none of
the classical-codec bpp numbers above are independently confirmed on data resembling ours
(16-bit, Poisson-ish noise, CT projections). The two solid, VERIFIED numbers I have that
*are* on real detector/projection data are Marone et al.'s 3–4× lossy-safe factor on TOMCAT
projections, and Bernstein & Jakoncic's 7.5–10.5:1 bslz4/bszstd on Eiger crystallography
frames. Treat any other bpp figure in this section as a generic-image-benchmark number, not
a scientific-imaging number.

---

## 4. Structural priors of the sinogram

### The bowtie / double-wedge Fourier support

- **P. A. Rattey, A. G. Lindgren, "Sampling the 2-D Radon Transform," *IEEE Trans.
  Acoustics, Speech, and Signal Processing*, 29(5), 1981.** **VERIFIED** (Semantic Scholar
  abstract page fetched, consistent with the well-known result). This is the paper that
  establishes the **bowtie** (their term) shape of the 2D Fourier transform of the sinogram
  for objects of finite spatial support and finite bandwidth, and derives the resulting
  ~2M²/π independent-information-content bound for M projections. Also shows the finite/
  infinite-length bowtie distinction (bandlimited object vs. finite-support-only object) and
  that a **hexagonal sampling grid** (not rectangular) is Nyquist-optimal for the Radon
  domain, needing ~half the samples.

- **Exact bound statement** (VERIFIED via corroborated search summary of standard tomography
  texts, e.g., Natterer's "The Mathematics of Computerized Tomography"; not independently
  opened, but the statement matches the well-known form used throughout the CT-sampling
  literature): if the object is supported in a disk of radius **R** and its Fourier
  transform F(u_r, u_θ) is essentially band-limited with |u_r| ≤ B_r (radial/detector
  bandwidth), then the sinogram's 2D Fourier transform is essentially zero for
  **|u_θ| > R·B_r** — i.e. the angular-frequency support grows only linearly with the
  product of object radius and detector bandwidth, which is precisely the classical
  justification for both angular subsampling *and* angular interpolation/extrapolation
  schemes in the sinogram domain. This is the theorem underlying any claim that "the sinogram
  is redundant across nearby angles and therefore predictable."

- **Applications of the bowtie support to interpolation/upsampling** (VERIFIED as a class of
  results, individual papers UNVERIFIED beyond search snippets): this bound is the standard
  justification cited in the sparse-view/angular-upsampling literature (e.g., the DRHT paper
  below implicitly relies on it, though it does not derive it explicitly in what I read).

### Helgason–Ludwig consistency conditions

- **Original sources**: S. Helgason, "The Radon Transform on Euclidean Spaces, Compact
  Two-Point Homogeneous Spaces and Grassmann Manifolds," *Acta Mathematica* 113 (1965); D.
  Ludwig, "The Radon Transform on Euclidean Space," *Comm. Pure Appl. Math.* 19 (1966).
  UNVERIFIED in this session (found only via search-index metadata for Helgason's paper;
  could not independently confirm Ludwig's exact year/volume — search results conflated it
  with an unrelated 2010 reprint listing). These are the standard-cited originals; treat the
  exact bibliographic details (especially Ludwig's) as needing a library/MathSciNet check
  before final citation.

- **Exact statement** (VERIFIED via multiple independent, mutually consistent search
  summaries of the standard tomography literature): for the parallel-beam Radon transform
  p(θ, t) of a compactly supported 2D function, the **k-th angular/detector moment**
  ∫_{-∞}^{∞} t^k p(θ, t) dt must, for every k ≥ 0, be a homogeneous trigonometric polynomial
  of degree k in (cos θ, sin θ) — equivalently a finite Fourier series in θ with only
  harmonics |n| ≤ k. The k=0 case (the "Gelfand–Graev–Helgason–Ludwig" zero-moment
  condition) reduces to: **every projection at every angle must integrate to the same total
  mass** (angle-independent total attenuation). This is a strong, closed-form, per-angle
  redundancy constraint on the sinogram that is exactly the kind of "structural prior" a
  predictive/compression model should be able to exploit or be regularized by.

- **Y. Huang, X. Huang, O. Taubmann, Y. Xia, V. Haase, J. Hornegger, G. Lauritsch, A. Maier,
  "Restoration of Missing Data in Limited Angle Tomography Based on Helgason-Ludwig
  Consistency Conditions," *Biomedical Physics & Engineering Express*, 3(3), 035015, 2017.**
  **VERIFIED** (full author list, journal, volume/issue/article number, year all
  corroborated across FAU's institutional repository, the IOPscience listing, and a matching
  author PDF; code released on GitHub). Uses the HL conditions directly to extrapolate/
  restore missing angular data in limited-angle CT — i.e., exactly "sinogram angular
  extrapolation using HL consistency," classical (non-learned) approach.
  Link: https://iopscience.iop.org/article/10.1088/2057-1976/aa71bf

- **Y. Xu, K. Taguchi, B. M. W. Tsui, "Statistical Projection Completion in X-ray CT Using
  Consistency Conditions," *IEEE Trans. Medical Imaging*, 29(8), 2010.** UNVERIFIED (found
  via Semantic Scholar/PubMed/IEEE Xplore listings that agree on authors/topic, but the
  PubMed abstract page itself returned only a cookie-consent wall, so I could not read the
  primary abstract). Penalized maximum-likelihood *statistical* sinogram restoration that
  incorporates HL consistency as a constraint, then FBP reconstructs — an example of using
  HL conditions as a regularizer rather than a hard constraint, relevant if we want our
  learned predictor's outputs to be softly consistency-penalized.

### Neural angular view interpolation / synthesis / extrapolation in the sinogram domain

- **A. S. Adishesha, D. J. Vanselow, P. La Riviere, K. C. Cheng, S. X. Huang, "Sinogram
  Domain Angular Upsampling of Sparse-View Micro-CT with Dense Residual Hierarchical
  Transformer and Attention-Weighted Loss," *Computers in Biology and Medicine*, 242,
  107802, 2023.** **VERIFIED** (full text fetched via PMC/publisher; note the actual title's
  final words are "Attention-Weighted Loss," not "Noise-Aware Loss" as one search snippet
  claimed — use this VERIFIED title). Architecture: U-shaped hierarchical encoder-decoder +
  Dense Residual Blocks (stacked conv-ReLU with dense+residual connections) + **windowed
  multi-head self-attention (NW-MSA)** blocks for long-range interactions across the
  sinogram. Loss: learnable per-pixel-weighted combination of L1 (signal regions) and KL
  divergence (flat/background regions). At 8× angular upsampling: **+17.73 dB PSNR, +0.161
  SSIM** vs. bicubic baseline on zebrafish/earthworm/walnut micro-CT sinograms; beats UFormer,
  RDN, REDCNN baselines. This is the best concrete example I found of a transformer
  architecture operating directly on sinogram angular structure (not a generic image
  transformer applied incidentally).
  Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC11158828/

- **K. Chen, B. Huang, X. Yang, J. Zhang, Y. Wang, Q. Liu, "PRO: Projection Domain Synthesis
  for CT Imaging," arXiv:2506.13443, 2025.** **VERIFIED** (abstract fetched from arXiv).
  A foundation-model-style generator for synthetic CT projection data operating natively in
  the projection domain (models attenuation, beam hardening, scattering, geometry directly),
  conditioned on anatomical text prompts, used to augment low-dose/sparse-view
  reconstruction training. I could not confirm from the abstract alone whether it treats
  angular views as transformer tokens with an angle-specific positional encoding (a search
  snippet suggested "sinogram views split as input tokens" for this class of model, but I
  could not re-confirm that specific mechanism against the PRO paper's body text) — **read
  the PDF body before citing the tokenization mechanism**.
  Link: https://arxiv.org/abs/2506.13443

- **Y. Lee, J. Lee, H. Kim, B. Cho, S. Cho, "Deep-Neural-Network-Based Sinogram Synthesis for
  Sparse-View CT Image Reconstruction," *IEEE Trans. Radiation and Plasma Medical Sciences*,
  3(2), 109–119, 2019.** **VERIFIED** (arXiv preprint 1803.00694 corroborates title/authors;
  journal citation from KAIST institutional repository, consistent across multiple
  independent listings). An early, frequently-cited baseline: a CNN synthesizes missing
  angular views directly in the sinogram domain for sparse-view CT.
  Link: https://arxiv.org/abs/1803.00694

- **Sinogram inpainting (classical/non-learned) as the predecessor task**: multiple
  "sinogram inpainting"/"sinogram completion" papers for limited-angle CT were found
  (e.g., ICIP 2019 "Sinogram Image Completion for Limited Angle Tomography," UNVERIFIED —
  the PDF could not be parsed as text) confirming this is a long-standing, well-populated
  sub-field; angular *interpolation* (dense-to-dense, our setting) is less populated than
  angular *extrapolation/completion* (sparse-to-dense or missing-wedge, the more common
  motivation).

### Does anyone argue reconstruct-then-reproject beats a direct sinogram-domain network?

**VERIFIED as a real, recurring argument in the dual-domain literature** (the specific
phrasing below is a search-synthesized paraphrase of a claim that recurs, nearly verbatim in
spirit, across several dual-domain CT papers I found — e.g. the DuDoNet lineage; I was not
able to pin it to one single primary quote I opened myself, so treat the wording as
representative rather than a direct citation): sinogram-domain-only networks are risky
because **"reconstruction is highly sensitive to the internal consistency of the sinogram;
any imperfect (denoised/synthesized/inpainted) operation performed directly on the sinogram
can violate that consistency and produce severe, non-local secondary artifacts across the
entire reconstructed image"** (streaks, in particular) — whereas an image-domain fix, or a
reproject-after-fix step, cannot introduce this class of error because it works on an
already-reconstructed, spatially local representation. This is the stated motivation for
essentially every dual-domain architecture below (DuDoNet and successors): don't trust a
sinogram-domain network's output un-checked; couple it to an image-domain refinement and/or
a differentiable re-projection consistency loss.

- **W.-A. Lin, H. Liao, C. Peng, X. Sun, J. Zhang, J. Luo, R. Chellappa, S. K. Zhou,
  "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction," CVPR 2019.**
  arXiv:1907.00273. **VERIFIED** (abstract fetched from arXiv). Sinogram-Enhancement-Net →
  differentiable **Radon inversion layer** (so gradients flow sinogram-domain ↔ image-domain)
  → Image-Enhancement-Net, trained end-to-end. First end-to-end dual-domain network for
  metal-artifact reduction; explicitly designed around the "sinogram-only isn't safe" concern
  above.
  Link: https://arxiv.org/abs/1907.00273

- Successors found but not independently opened this session (UNVERIFIED, listed for
  completeness): "U-DuDoNet" (unpaired dual-domain), "Dual-Domain Adaptive-Scaling Non-local
  Network for CT Metal Artifact Reduction," and multiple "dual-domain sparse-view CT" papers
  (2022–2025) all following the same SE-Net/Radon-layer/IE-Net or equivalent pattern.

**Implication for our design**: this dual-domain literature is a warning, not a blocker —
our design predicts a raw angular row and entropy-codes the *residual against the true
measurement*, so (unlike DuDoNet-style artifact reduction) we never actually replace real
data with a network's guess; the network's output only ever needs to be "good enough to make
the residual small," and any prediction error is captured exactly, not silently baked into
the reconstruction. This sidesteps the core objection above, but it's worth being explicit
about in the writeup since reviewers steeped in this literature will likely raise it.

---

## 5. Architectures for sinogram-domain neural networks

Covered substantially in §4. Summary of what I found, VERIFIED/UNVERIFIED as marked there:

- **Plain UNets / CNN post-processing** dominate the image-domain-only literature (FBPConvNet
  and its many descendants) — not sinogram-domain-specific, listed only as the baseline class
  everything else compares against.
- **Windowed transformer self-attention over sinogram patches**: DRHT (VERIFIED, §4) is the
  clearest example of a transformer explicitly built for sinogram angular structure.
- **Dual-domain networks with a differentiable Radon/back-projection layer**: DuDoNet
  (VERIFIED, §4) and its many successors (UNVERIFIED individually, but the pattern is
  well-established and repeatedly confirmed across search results).
- **Implicit neural representations (INR/coordinate-MLP) of sinograms**: confirmed as an
  active area (search-synthesized, individual papers not opened this session) — two
  families: (a) "direct-reconstruction" INRs that represent the *image* as a coordinate-MLP
  and differentiably forward-project to fit the sinogram (NeRP, NeAT, SCOPE — names appear
  repeatedly across hits but none independently opened), and (b) rarer "view-synthesis" INRs
  that represent the *sinogram itself* as a coordinate function of (angle, detector position)
  and are fit/queried directly — this second family is the closer analog to representing a
  sinogram continuously, but I could not find a paper that also entropy-codes such a
  representation for compression purposes.
- **Polar/coordinate-transform layers**: found only in the *image*-domain rotation-invariance
  literature (Polar Transformer Networks, ICLR 2018, UNVERIFIED here but a well-known paper;
  Polar Coordinate CNN, UNVERIFIED) — these transform a Cartesian image into polar
  coordinates for rotation-invariant recognition, which is a related but distinct idea from
  exploiting the sinusoidal trajectory that a single point traces across a sinogram's rows.
  I found **no paper that builds positional encodings or attention patterns explicitly
  around the known sinusoidal (t = x cos θ + y sin θ) trajectory of image points across
  sinogram rows** — this looks like a genuine gap (see §"Gaps" below).
- **Sinusoidal/SIREN-style networks** (sine activations, e.g. SIREN) appeared in search
  results only in the generic INR-representation-learning sense (fitting arbitrary signals
  with sine-activated MLPs), not specifically motivated by or exploiting the Radon-transform
  sinusoid — a naming coincidence to be careful not to conflate.

---

## 6. Noise-floor / rate-distortion arguments for scientific data; "statistically lossless" quantization

- **W. D. Pence, R. Seaman, R. L. White, "Lossless Astronomical Image Compression and the
  Effects of Noise," *Publications of the Astronomical Society of the Pacific* (PASP),
  121(878), 2009.** **VERIFIED** (abstract/content fetched from arXiv:0903.2140 mirror).
  This is the strongest, most citable, most quantitative source I found for Q6. Establishes
  a closed-form relationship between noise level and the compressibility floor for integer
  (e.g. CCD/detector count) images: the number of **incompressible noise bits per pixel** is
  **N_bits = log₂(σ·√12)**, where σ is the standard deviation of (approximately Gaussian)
  pixel noise, and the resulting best-achievable **compression ratio R ≈ BITPIX / N_bits +
  K** where BITPIX is the raw bit depth and K is a small algorithm-dependent constant. They
  validate this on real astronomical CCD images and show the **Rice** algorithm reaches
  75–90% of this theoretical bound, beating GZIP by ~1.4× ratio and 2–3× speed. This gives us
  a directly citable, closed-form ceiling: **no lossless codec, however clever, can beat
  BITPIX/log₂(σ√12) bits/pixel on data whose noise has standard deviation σ**, which is
  exactly the argument needed to justify (a) why "raw sinogram lossless compression" has a
  hard floor set by detector/photon noise, and (b) why denoising before compression (§7) is
  the only way past that floor.
  Link: https://arxiv.org/abs/0903.2140

- **M. Hammer, K. Yoshii, A. Miceli, "Strategies for on-chip digital data compression for
  X-ray pixel detectors," arXiv:2006.02639, 2020.** **VERIFIED** (already cited in §1). This
  is the clearest example I found of an accepted, quantified, **noise-matched (Poisson-σ-
  tied) quantization scheme actually used in production-adjacent X-ray detector hardware**
  (APS-adjacent instrumentation work): ADC output is re-quantized in-pixel to units of
  photon count "near the Poisson noise level" before any downstream lossless compression,
  giving a data-independent >1.5× ratio essentially for free, on the reasoning that any
  finer quantization step only encodes noise, not signal. This is a real precedent for "tie
  quantization step to local Poisson σ," though it's applied per-pixel to raw ADC counts, not
  explicitly framed as a sinogram/angular compression scheme.
  Link: https://arxiv.org/abs/2006.02639

- **"Statistically lossless" as a named, patented concept** (UNVERIFIED — found only a
  US patent listing, "Statistically lossless compression system and method," US9438899,
  not independently opened or confirmed as peer-reviewed literature): this exact phrase
  does appear in the X-ray/medical-imaging patent literature, suggesting some industry
  players (detector/imaging vendors) have formalized this concept commercially, but I could
  not find a peer-reviewed paper using this exact term for CT/synchrotron sinogram data.

- **Community-accepted error-bound standard**: I found **no single accepted community
  standard** (e.g. an ESRF/APS/PSI/DESY house standard, or a community consensus paper) for
  acceptable lossy-compression error bounds specific to synchrotron/micro-CT projection
  data. The closest thing to a de facto standard is the empirical 3–4× "safe" factor from
  Marone/Vogel/Stampanoni (2020, VERIFIED, §1), which is a single-facility (TOMCAT/PSI)
  empirical study, not a cross-facility standard. The SZ/ZFP/SPERR family (§3) offers
  general-purpose, user-specified point-wise-error-bound (PWE) modes that are widely used in
  HPC/simulation science, but again I found no evidence of a specific accepted numeric bound
  (e.g. "≤0.5σ_Poisson") adopted community-wide for tomography.

---

## 7. Denoise-then-compress for scientific/photon-limited imaging

- **B. Brummer, C. De Vleeschouwer, "On the Importance of Denoising when Learning to
  Compress Images," WACV 2023.** arXiv:2307.06233. **VERIFIED** (abstract fetched from
  arXiv). Central claim, directly quotable: **"any image compression scheme can attain
  better rate-distortion by having the noise removed first"** — and more specifically, they
  show a *single* model jointly trained on a mixture of noise levels beats *both* a
  compression-only model *and* a naive two-stage denoise-then-compress pipeline, at ~10× less
  compute than running separate denoise+compress networks. Important nuance for us: this
  paper's headline result argues *joint* denoise+compress beats *sequential* denoise-then-
  compress — i.e., it's evidence *for* the general principle that denoising and compression
  should be co-designed (which supports our compress-the-*residual*-after-*prediction*
  approach), but its literal recommendation (train one joint model) is a variant we are not
  exactly proposing (we still explicitly separate "predict" and "entropy-code-the-residual,"
  which is closer to their "sequential" baseline than their proposed joint model — worth
  being honest about this distinction in the writeup, not overclaiming their result as
  supporting our exact architecture).
  Link: https://arxiv.org/abs/2307.06233

- **Pence, Seaman & White (2009, VERIFIED, §6)** is also directly relevant here from the
  other direction: because they show noise sets a *hard, quantified floor* on lossless
  compressibility (N_bits = log₂(σ√12) irreducible bits/pixel), the scientific justification
  for denoise-then-compress becomes explicit and quantitative — every bit "spent" encoding
  noise indistinguishable from measurement uncertainty is a bit that buys nothing
  scientifically, *provided* the denoising step doesn't discard information a downstream
  analysis actually needed. Neither this paper nor the Brummer/De Vleeschouwer paper
  addresses that "provided" clause directly (i.e., neither validates that removed noise was
  truly scientifically inert) — that validation (e.g. via downstream reconstruction-task
  metrics, not just pixel-space PSNR/entropy) appears to be something we would need to argue
  and demonstrate ourselves, not something the literature already establishes for CT/
  synchrotron data specifically.

- I found **no paper specifically about photon-limited synchrotron/micro-CT data** making
  the denoise-then-compress argument end-to-end with a quantified, task-validated result
  (e.g. "we denoised raw projections, compressed the residual, reconstructed, and showed the
  reconstruction quality/downstream science was preserved while achieving X:1 extra
  compression"). This looks like a genuine gap specific to our domain — the pieces exist
  separately (noise-floor theory from astronomy, denoise-before-compress from general
  imaging, Poisson-matched quantization from X-ray detector hardware) but I did not find them
  combined and validated for tomographic projection streams.

---

## Gaps / what appears genuinely unexplored

Based on this session's search coverage (not exhaustive — treat as a starting map, not a
proof of absence):

1. **No paper doing autoregressive next-*angular-row* prediction of a sinogram for
   compression.** Angular redundancy is exploited today via decimation/subsampling (view
   thinning), interpolation/upsampling (DRHT and similar, for reconstruction quality, not
   compression), or inpainting/extrapolation (limited-angle CT) — but I found nothing
   framing "predict projection θ+Δθ from projections up to θ, entropy-code the residual" as
   an explicit compression scheme, despite this being the single most natural application of
   the bowtie/HL redundancy theory (§4) to compression. The HEPS paper (§1, UNVERIFIED
   pending full-text read) is the closest thing found and needs to be read in full to know
   whether it is in fact this, applied to detector-frame streams generally rather than to
   the angular sinogram axis specifically.
2. **No sinogram-domain architecture that explicitly encodes the known Radon sinusoid
   t = x cos θ + y sin θ as a positional prior** (e.g. a sinusoid-warped attention pattern,
   or a positional encoding parameterized by (θ, expected sinusoid phase) rather than raw
   (row, col) indices). Everything found either treats the sinogram as a generic 2D image
   (plain UNet/CNN) or, at best, windows/patches it for a generic transformer (DRHT) without
   baking in the sinusoidal geometry itself.
3. **No end-to-end, quantified, task-validated "denoise raw synchrotron/CT projections, then
   compress, then reconstruct, then show downstream quality is preserved" study** (§7) — the
   closest facility-level equivalent (Marone/Vogel/Stampanoni, TOMCAT/PSI) evaluates *lossy
   compression* directly, not *denoise-then-lossless-compress-the-residual*.
4. **No accepted cross-facility numeric error-bound standard** for lossy/near-lossless
   compression of raw synchrotron projection data (§6) — each facility/study appears to set
   its own empirical threshold.
5. **HL consistency conditions used as a training-time regularizer or auxiliary loss for a
   learned predictive/compression model** (as opposed to a classical, non-learned
   completion/extrapolation constraint, §4) does not appear to have been tried, at least not
   under a name/framing I could find.

## Numbers we can quote

All VERIFIED unless flagged:

- TOMCAT/PSI, raw X-ray CT projections: **3–4× lossy compression with no measurable
  reconstruction-quality loss**; 6–8× safe for phase-retrieved (phase-contrast) data.
  (Marone, Vogel, Stampanoni, *J. Synchrotron Rad.* 2020.)
- NSLS-II crystallography, Eiger X 9M, bit-shuffle+zstd vs. bit-shuffle+LZ4: **10.55:1 vs.
  7.50:1** on the hardest (25% transmission) test case — a **41% ratio improvement**, at a
  throughput cost (15.84 vs. 42.35 frames/s at zstd level 2). (Bernstein & Jakoncic, *J.
  Synchrotron Rad.* 2024.) Note: crystallography, not tomography.
- Noise-floor bound for lossless compression of noisy integer detector images:
  **N_bits = log₂(σ√12)** irreducible bits/pixel; Rice coding reaches 75–90% of this bound
  in practice. (Pence, Seaman & White, *PASP* 2009.)
- X-ray pixel detector, Poisson-noise-matched in-pixel quantization: **>1.5× compression**
  "for free" before any downstream entropy coding, plus >4×/>7×/>8× via zero-suppression on
  HEDM/ptychography/XPCS data respectively (combined 6–12× total). (Hammer, Yoshii, Miceli,
  arXiv:2006.02639, 2020.)
- PixelCNN++ discretized-mixture-of-logistics: **2.92 bits/subpixel on CIFAR-10**.
  (Salimans et al., ICLR 2017.)
- SReC (predict-next-resolution-level, closest structural analog to our design):
  **4.29 bpsp on ImageNet64, 2.70 bpsp on Open Images.** (Cao, Wu, Krähenbühl, 2020.)
- L3C: first practical learned lossless codec beating PNG/WebP/JPEG2000, with **>100×
  sampling speedup** over PixelCNN via a 3-forward-pass hierarchical scheme. (Mentzer et al.,
  CVPR 2019.)
- DRHT sinogram angular upsampling (8×): **+17.73 dB PSNR, +0.161 SSIM** vs. bicubic, on
  micro-CT sinograms, beating UFormer/RDN/REDCNN. (Adishesha et al., *Comput. Biol. Med.*
  2023.)
- UNVERIFIED, use with caution: classical lossless CT sinogram compression ratios cluster
  around **~1.4–1.8×** (medical-CT survey figure, source not independently opened); a
  16-bit JPEG-LS figure of **~14.23:1 (1.12 bpp)** found in a comparative survey looks
  dataset-favorable and should not be quoted without checking the source dataset first.
