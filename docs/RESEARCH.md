# Background & Research — Image Palette Extractor

This document records the algorithms and colour science the
implementation is based on, and states precisely where the from-scratch
code follows the canonical method and where it simplifies it. Every
source below was consulted while writing the code.

## 1. The clustering problem and Lloyd's algorithm

Grouping pixels into *k* dominant colours is the **k-means problem**:
choose *k* centres μ₁…μ_k minimising the objective (called **inertia**
in scikit-learn's terminology)

    J = Σᵢ ‖xᵢ − μ_{c(i)}‖²

where c(i) is the nearest centre to point xᵢ [1][2]. The problem is
NP-hard in general; the standard heuristic is **Lloyd's algorithm**,
proposed by Stuart Lloyd at Bell Labs in 1957 for pulse-code
modulation quantisation and published as "Least squares quantization
in PCM" (*IEEE Transactions on Information Theory*, 1982) [3][4]. It
alternates:

1. **Assignment** — assign each point to its nearest centre (inducing a
   Voronoi partition of the space);
2. **Update** — move each centre to the mean of its assigned points;

until the centres stop moving. Both steps never increase J, so the
iteration converges — but only to a **local** optimum, which makes the
initialisation matter (see §2, and the worked example in the README,
where the default RGB run lands in a visibly worse local optimum than
the Lab run). Edward Forgy published essentially the same method in
1965, hence "Lloyd–Forgy algorithm" [4].

This implementation follows Lloyd's algorithm directly, including:

- convergence when the largest centre movement in an iteration drops
  below a tolerance (`tol=1e-4`, mirroring scikit-learn's parameter
  of the same name) or `max_iter` is reached [5];
- reporting the final **inertia**, recomputed from the returned labels
  and centres, exactly as `KMeans.inertia_` is defined — "sum of
  squared distances of samples to their closest cluster center" [5];
- empty-cluster handling: a centre that loses all points is restarted
  at the point worst served by the current centres.

### Documented simplification: no `n_init`

scikit-learn runs the algorithm `n_init` times with different seeds
and keeps the lowest-inertia result (with `init='k-means++'` the
modern default is a single run) [5]. This tool deliberately performs
**one** run per invocation with an explicit `--seed`, so results are
exactly reproducible from the command line; users who want best-of-N
behaviour can run several seeds and compare the reported inertia.

## 2. Initialisation: k-means++

**k-means++** (Arthur & Vassilvitskii, SODA 2007) seeds the centres
from the data itself [6][7]:

1. Choose the first centre uniformly at random from the data points.
2. Let D(x) be the distance from x to the nearest centre already
   chosen. Choose each subsequent centre from the data with
   probability proportional to **D(x)²** ("D² weighting").
3. Run Lloyd's algorithm from those centres.

The paper proves this seeding alone is Θ(log k)-competitive with the
optimal clustering, and that it improves Lloyd's algorithm in both
speed and accuracy in practice [6][8]. `kmeans_plusplus_init` in
`src/palette/kmeans.py` implements exactly these steps (the squared
distances are maintained incrementally: after each new centre, the
running minimum distance is updated). `--init random` is provided for
comparison, matching the other classical option.

## 3. Colour spaces: why cluster in Lab?

RGB coordinates are not perceptually uniform: a fixed Euclidean step
in RGB corresponds to very different perceived colour differences
depending on where in the cube it sits, and the channels are strongly
correlated with brightness. **CIE L\*a\*b\*** (CIELAB, CIE 1976) was
designed so that Euclidean distance approximates perceived difference
(the CIE76 ΔE\*_ab metric is literally the Euclidean distance in Lab).
So in `--space lab` mode the tool clusters perceptual coordinates,
which in practice separates small but vivid regions (the sample's sun)
that RGB clustering absorbs — see the measured comparison in the
README.

### Conversion chain implemented in `colorconv.py`

sRGB → Lab follows the standard chain (constants as published by
Bruce Lindbloom [9][10][11]):

1. **Gamma decode** each channel c ∈ [0,1]:
   c_lin = c/12.92 if c ≤ 0.04045, else ((c + 0.055)/1.055)^2.4.
2. **Linear RGB → CIE XYZ** (sRGB primaries, D65) with the matrix

       [0.4124564  0.3575761  0.1804375]
       [0.2126729  0.7151522  0.0721750]
       [0.0193339  0.1191920  0.9503041]

3. **XYZ → Lab** against the D65 reference white
   (Xn, Yn, Zn) = (0.95047, 1.0, 1.08883), with

       f(t) = ∛t            if t > (6/29)³
            = ( (29/3)³·t + 16 ) / 116   otherwise,
       L* = 116·f(Y/Yn) − 16
       a* = 500·( f(X/Xn) − f(Y/Yn) )
       b* = 200·( f(Y/Yn) − f(Z/Zn) )

The inverse path (Lab → XYZ → linear RGB → gamma encode, with the
published inverse matrix) converts cluster centres found in Lab back
to displayable sRGB; out-of-gamut centre components are clipped, as
documented in the code. The conversion tests check the implementation
against published reference values — pure red maps to
L\*a\*b\* ≈ (53.24, 80.09, 67.20) — and verify a lossless round trip
(≤ 1 unit per channel after rounding) over random colours [9].

### Simplifications relative to full colour science

- Distance in Lab is **CIE76** (plain Euclidean). The more accurate
  CIEDE2000 ΔE formula exists but is not a metric on coordinates, so
  it cannot drive a centroid-based algorithm; CIE76 is the coherent
  choice for k-means.
- Pixels are weighted equally; there is no saliency/centre weighting.
- Very large images are evenly sub-sampled (deterministic indices) to
  a pixel budget rather than clustered exhaustively.

## References

1. "A brief history of K-means" — A. N. K. (andrij.me): Lloyd's PCM
   motivation and the objective φ = Σᵢ min_j ‖xᵢ − μ_j‖².
   https://andrij.me/posts/a-brief-history-of-kmean
2. scikit-learn. `sklearn.cluster.k_means` documentation — inertia
   definition, `init='k-means++'`, `n_init` best-of behaviour.
   https://scikit-learn.org/1.2/modules/generated/sklearn.cluster.k_means.html
3. S. P. Lloyd. "Least squares quantization in PCM." *IEEE
   Transactions on Information Theory* 28(2), 129–137, 1982.
   DOI 10.1109/TIT.1982.1056489 (full citation and history via [4]).
4. "Lloyd's algorithm." StarOcean's Wiki (1957 Bell Labs origin,
   1982 publication, Lloyd–Max/Forgy notes, references).
   http://www.staroceans.org/wiki/A/Lloyd's_algorithm
5. scikit-learn. `sklearn.cluster.KMeans` documentation — parameter
   semantics (`tol`, `max_iter`, `inertia_`) mirrored by this tool.
   https://scikit-learn.org/0.21/modules/generated/sklearn.cluster.KMeans.html
6. D. Arthur and S. Vassilvitskii. "k-means++: The Advantages of
   Careful Seeding." *Proc. 18th ACM-SIAM Symposium on Discrete
   Algorithms (SODA)*, 2007, pp. 1027–1035. ACM Digital Library:
   https://dl.acm.org/doi/10.5555/1283383.1283494
7. Arthur & Vassilvitskii, paper PDF (author copy, JHU course mirror):
   https://www.cs.jhu.edu/~ayuille/JHUcourses/VisionAsBayesianInference2022/3/K-means++_Arthur_Vassilvitskii.pdf
8. "k-means++." Wikipedia (D²-weighting procedure summary).
   https://en.wikipedia.org/wiki/k-means++
9. P. G. Jamadagni. `rgb2lab.c` — sRGB/XYZ/Lab conversion code whose
   constants and matrices are taken from Bruce Lindbloom's reference
   (sRGB→XYZ matrix, D65 white (0.95047, 1, 1.08883), ε/κ constants).
   https://github.heygears.com/jamadagni/rgb2lab/blob/master/rgb2lab.c
10. P. Occil. "Color Topics for Programmers" (XYZToLab/LabToXYZ
    pseudocode following Lindbloom's ε = 216/24389, κ = 24389/27).
    https://peteroupc.github.io/colorgen.pdf
11. `rgbxyzlab` package notes (PyPI) — "most of the math used to
    implement the conversions is from Bruce Lindbloom's webpage"
    (independent confirmation of the same constants).
    https://pypi.org/project/rgbxyzlab/
