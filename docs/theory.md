# Orientation Codec: Theory and Methods

## 1. Introduction

Generative neural networks operating on microstructure images require
input–output representations that are both *continuous* and
*compatible with standard vision architectures*. Crystallographic
orientations, conventionally expressed as Bunge Euler angles
$(\varphi_1, \Phi, \varphi_2)$ in the $ZXZ$ convention, violate both
requirements: the angular coordinates are periodic (introducing
discontinuities at $0/2\pi$ wraparound), and the mandatory symmetry
folding into a fundamental zone (FZ) creates artificial high-frequency
edges between physically similar grains. This document describes the
mathematical framework of the orientation codec — a bijective mapping
from the space of crystallographic orientations to three-channel RGB
images and back — that resolves these issues while preserving
sub-degree fidelity.

---

## 2. Orientation Representation

### 2.1. Quaternion Parameterisation

Each pixel-wise orientation is first converted from Bunge Euler angles
to a unit quaternion $\mathbf{q} = (q_x, q_y, q_z, q_w) \in S^3$ via
the standard $ZXZ$ decomposition:

$$
\mathbf{q} = R_z(\varphi_1)\; R_x(\Phi)\; R_z(\varphi_2)
$$

where $R_z(\theta)$ and $R_x(\theta)$ are the elementary rotation
quaternions. Quaternions form a double cover of $SO(3)$: both
$\mathbf{q}$ and $-\mathbf{q}$ represent the same physical rotation.
We resolve this ambiguity by enforcing the hemisphere convention
$q_w \geq 0$ throughout.

### 2.2. HCP Crystal Symmetry

For hexagonal close-packed (HCP) crystals, the relevant symmetry group
is the proper rotation subgroup of point group $6/mmm$, denoted $622$
(Schoenflies: $D_6$), which has order 12. The 12 symmetry operators
$\{s_k\}_{k=1}^{12}$ consist of:

- **Six** rotations about the $c$-axis ($z$):

$$
s_{2k-1} = R_z\!\left(\frac{k\pi}{3}\right), \quad k = 0, 1, \ldots, 5
$$

- **Six** two-fold rotations — each $c$-axis rotation composed with a
  $180°$ rotation about the $x$-axis:

$$
s_{2k} = R_z\!\left(\frac{k\pi}{3}\right) \cdot R_x(\pi), \quad k = 0, 1, \ldots, 5
$$

Any orientation $\mathbf{q}$ is physically equivalent to all 12
symmetry-related orientations $\{s_k \cdot \mathbf{q}\}_{k=1}^{12}$.
This equivalence is central to both the encoding (Section 3) and
decoding (Section 4) pipelines.

---

## 3. Encoding Pipeline (Dream3D $\rightarrow$ RGB)

Given a Dream3D microstructure file containing a grain label map
$L \in \mathbb{Z}^{H \times W}$ and per-pixel Euler angles, the
encoding proceeds in five steps.

### 3.1. Per-Grain Quaternion Extraction

For each grain $g$ with pixel set
$\Omega_g = \{(r,c) : L(r,c) = g\}$, a single representative
quaternion $\mathbf{q}_g$ is extracted from a pixel in $\Omega_g$
(all pixels within a grain carry identical orientations in the
synthetic RVE data).

### 3.2. Continuous Unfolding via Anchored BFS

**Problem.** The conventional approach of folding each orientation
into the fundamental zone independently (i.e., selecting the
symmetry equivalent $s_k \cdot \mathbf{q}_g$ with the largest $q_w$)
produces artificial colour discontinuities. Two adjacent grains with a
physical misorientation of only $2°$ may be mapped to opposite ends of
quaternion space after independent FZ folding.

**Solution.** We perform a breadth-first search (BFS) on the grain
adjacency graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where
vertices $\mathcal{V}$ are grains and edges
$\mathcal{E} = \{(g_i, g_j) : \Omega_{g_i} \text{ and } \Omega_{g_j}
\text{ are 4-connected neighbours}\}$. The algorithm is as follows:

1. **Initialise** the BFS root as the grain with the largest pixel
   count. If an anchor quaternion $\mathbf{q}_{\mathrm{anchor}}$ is
   provided (see Section 3.3), the root is first aligned to it by
   selecting:

$$
\tilde{\mathbf{q}}_{\mathrm{root}} = \arg\max_{s_k,\;\sigma \in \{+1,-1\}} \;
\sigma \cdot s_k \cdot \mathbf{q}_{\mathrm{root}} \cdot \mathbf{q}_{\mathrm{anchor}}
$$

   where the dot product is the 4D inner product.

2. **Propagate.** For each unvisited neighbour $g_j$ of the current
   grain $g_i$, choose the symmetry equivalent and sign that maximises
   the quaternion dot product with the parent:

$$
\tilde{\mathbf{q}}_{g_j} = \arg\max_{s_k,\;\sigma \in \{+1,-1\}} \;
\left\langle \tilde{\mathbf{q}}_{g_i},\;
\sigma \cdot s_k \cdot \mathbf{q}_{g_j} \right\rangle
$$

This greedy local alignment propagates through the entire adjacency
graph, yielding an orientation field where the quaternion distance
between neighbouring grains equals the true physical misorientation
angle — no artificial jumps.

**Complexity.** Each grain–neighbour pair requires $2 \times 12 = 24$
dot products. For $N$ grains the total cost is $O(|\mathcal{E}|)$,
which is linear in the number of grain boundaries.

### 3.3. Class-Level Global Frame Centering

Rather than computing a per-sample mean (which would require storing a
separate metadata file for each image at inference time), a single
**class-level** mean quaternion $\bar{\mathbf{q}}_c$ is pre-computed
for each alloy class $c$ by averaging over a representative
subsample.

**Mean quaternion computation.** Given $M$ file-level mean quaternions
$\{\mathbf{m}_j\}_{j=1}^{M}$ (each itself the mean over all grains
in one Dream3D file), the class mean is computed via the eigenvalue
method of Markley et al. (2007):

$$
\bar{\mathbf{q}}_c = \text{leading eigenvector of} \quad
\mathbf{A} = \frac{1}{M} \sum_{j=1}^{M} \mathbf{m}_j \, \mathbf{m}_j^\top
$$

where $\mathbf{A} \in \mathbb{R}^{4 \times 4}$ is a symmetric
positive semi-definite matrix and the leading eigenvector (largest
eigenvalue) is the $L_2$-optimal average over the quaternion
double-cover.

**Centering.** All grain quaternions are shifted so that
$\bar{\mathbf{q}}_c$ maps to the identity:

$$
\mathbf{q}_g^{(\mathrm{c})} = \bar{\mathbf{q}}_c^{-1} \cdot \tilde{\mathbf{q}}_g
$$

This concentrates the quaternion distribution around
$\mathbf{q} = (0,0,0,1)$, placing it in the region of quaternion space
where the subsequent stereographic projection has the highest
precision and no sign-flip singularities.

### 3.4. Stereographic Projection to $\mathbb{R}^3$

Each centred quaternion
$\mathbf{q}^{(\mathrm{c})} = (q_x, q_y, q_z, q_w)$ with $q_w \geq 0$
is projected to a three-dimensional stereographic coordinate:

$$
\mathbf{S} = \frac{1}{1 + q_w}
\begin{pmatrix} q_x \\ q_y \\ q_z \end{pmatrix}
\in [-1, 1]^3
$$

**Properties:**

- **Smoothness.** The map is $C^\infty$ on the open hemisphere
  $q_w > 0$. After centering (Section 3.3), virtually all orientations
  satisfy $q_w \gg 0$, so the projection operates far from its pole.

- **Bounded range.** For the centred distribution, $\|\mathbf{S}\| \ll 1$,
  and values are tightly clustered in $[-1, 1]^3$.

- **No square roots.** Unlike $q_w = \sqrt{1 - q_x^2 - q_y^2 - q_z^2}$
  reconstruction, the inverse (Section 4.2) uses only rational
  arithmetic, making it robust to noise.

### 3.5. Quantisation and Image Output

The stereographic coordinates $\mathbf{S} \in [-1,1]^3$ are mapped to
integer pixel values in a three-channel RGB image.

**16-bit TIFF** (lossless mode):

$$
\mathrm{pixel}_i = \mathrm{round}\!\left(
\frac{S_i + 1}{2} \times 65535
\right), \qquad i \in \{R, G, B\}
$$

The quantisation step size is
$\Delta = 2 / 65535 \approx 3.05 \times 10^{-5}$, corresponding to an
angular precision of approximately $0.005°$.

**8-bit PNG** (recommended for pretrained vision models):

The 16-bit image is rescaled to 8 bits:

$$
\mathrm{pixel}_i^{(8)} = \mathrm{round}\!\left(
\mathrm{pixel}_i^{(16)} \times \frac{255}{65535}
\right)
$$

The quantisation step is $\Delta = 2 / 255 \approx 7.84 \times 10^{-3}$,
corresponding to a mean angular error of $\sim 0.6°$ and a worst-case
error of $\sim 1.4°$. This is far below the $5°$ grain-boundary
misorientation threshold and has no measurable effect on grain
structure metrics (boundary F1 > 0.9999, grain-size Pearson $r > 0.999$).

The PNG format yields images of $\sim 40$ KB versus $\sim 527$ KB for
TIFF, a $\sim 13 \times$ compression ratio, and is directly compatible
with pretrained ViT, ResNet, and CLIP encoders without format
conversion.

---

## 4. Decoding Pipeline (RGB $\rightarrow$ Dream3D)

### 4.1. Pixel Value Unpacking

For an image with pixel values $p \in [0, V_{\max}]$ (where
$V_{\max} = 65535$ for uint16 and $V_{\max} = 255$ for uint8), the
stereographic coordinates are recovered as:

$$
S_i = \frac{p_i}{V_{\max}} \times 2 - 1, \qquad i \in \{R, G, B\}
$$

### 4.2. Inverse Stereographic Projection

The unit quaternion is reconstructed from $\mathbf{S}$ via rational
functions (no square roots or trigonometric calls):

$$
\|\mathbf{S}\|^2 = S_x^2 + S_y^2 + S_z^2
$$

$$
q_w = \frac{1 - \|\mathbf{S}\|^2}{1 + \|\mathbf{S}\|^2}, \qquad
\mathbf{q}_{xyz} = \frac{2\,\mathbf{S}}{1 + \|\mathbf{S}\|^2}
$$

The resulting quaternion $\mathbf{q} = (\mathbf{q}_{xyz},\, q_w)$ is
normalised to unit length and flipped to the $q_w \geq 0$ hemisphere.

**Noise robustness.** As $\|\mathbf{S}\|^2 \to \infty$, the
reconstruction smoothly approaches $q_w \to -1$ rather than producing
NaN. A $2\%$ perturbation in pixel values induces approximately a
$2°$ orientation shift — graceful degradation with no catastrophic
failures.

### 4.3. Global Frame Restoration

The class mean quaternion $\bar{\mathbf{q}}_c$ (from
`class_means.json`) restores the original reference frame:

$$
\mathbf{q}_g^{(\mathrm{rec})} = \bar{\mathbf{q}}_c \cdot \mathbf{q}^{(\mathrm{c})}
$$

### 4.4. Symmetry Folding to Fundamental Zone

The recovered orientation must be returned to the standard fundamental
zone for crystallographic consistency. For each pixel, the symmetry
equivalent with the highest $q_w$ is selected:

$$
\mathbf{q}_g^{(\mathrm{FZ})} = \arg\max_{s_k \in \{s_1, \ldots, s_{12}\}}
\left[ s_k \cdot \mathbf{q}_g^{(\mathrm{rec})} \right]_w
$$

where $[\cdot]_w$ denotes the scalar (real) part of the quaternion
product, and the sign convention $q_w \geq 0$ is enforced after each
symmetry application.

This step is vectorised: all $N = H \times W$ pixels are processed
simultaneously for each of the 12 symmetry operators, yielding $O(12N)$
total operations with no Python-level loops.

### 4.5. Euler Angle Conversion and Dream3D Export

Quaternions are converted to Bunge Euler angles
$(\varphi_1, \Phi, \varphi_2)$ in the $ZXZ$ convention and reduced
modulo $2\pi$ to the standard range $[0, 2\pi)$. The resulting
orientation map, together with a grain label map (either the original
or one recovered by segmentation), is written as a Dream3D HDF5 file with
a companion XDMF descriptor for ParaView visualisation.

---

## 5. Grain Segmentation from Encoded Images

When no ground-truth label map is available (e.g., for images generated
by a neural network), the grain structure can be recovered directly
from the RGB image.

Two pixels $(r_1, c_1)$ and $(r_2, c_2)$ are assigned to the same
grain if they are 4-connected and their max-channel intensity
difference is below a tolerance $\tau$:

$$
\max_{i \in \{R,G,B\}} |p_i(r_1, c_1) - p_i(r_2, c_2)| \leq \tau
$$

where $\tau = 1$ for uint8 images and $\tau = 50$ for uint16 images.
Connected components are extracted via sparse-graph analysis
(`scipy.sparse.csgraph.connected_components`), yielding a 1-based
grain label map $L' \in \mathbb{Z}^{H \times W}$.

For clean encoder output, this recovers virtually identical grain
boundaries as the original label map (boundary F1 > 0.9999).

---

## 6. Quantisation Error Analysis

### 6.1. Angular Precision

The end-to-end quantisation error arises solely from the finite bit
depth of the image representation. Let $\Delta$ be the quantisation
step in stereographic space. For a centred quaternion near identity
($q_w \approx 1$, $\|\mathbf{S}\| \approx 0$), the angular error
$\delta\theta$ induced by a perturbation $\delta S$ is:

$$
\delta\theta \approx 2\,\|\delta\mathbf{S}\| \quad \text{(radians)}
$$

For 16-bit: $\delta S = 1/65535 \approx 1.5 \times 10^{-5}$, giving
$\delta\theta \approx 0.002°$ per channel ($\sqrt{3}\times$ for the
worst-case 3-channel diagonal error $\approx 0.004°$).

For 8-bit: $\delta S = 1/255 \approx 3.9 \times 10^{-3}$, giving
$\delta\theta \approx 0.45°$ per channel ($\sqrt{3}\times$ worst case
$\approx 0.78°$; observed mean $\approx 0.6°$, observed max $\approx 1.4°$).

### 6.2. Experimental Verification

Round-trip tests on three HCP magnesium alloy classes (ME21, Mg-10Gd,
AZ31 heat-treated) with $300 \times 300$ images and 257–2,044 grains
per sample yield:

| Metric | TIFF 16-bit | PNG 8-bit |
|--------|-------------|-----------|
| Boundary F1 | 0.9999–1.0000 | 0.9999–1.0000 |
| Grain-size Pearson $r$ | 0.9991–0.9997 | 0.9991–0.9997 |
| PNG-vs-TIFF quat. angle (mean) | — | 0.61°–0.69° |
| PNG-vs-TIFF quat. angle (max) | — | 1.30°–1.37° |
| Segmentation count accuracy | 102–103% | 102–103% |

The slight over-count in segmentation ($\sim 2$–$3\%$) arises from
sub-pixel boundary regions where adjacent grains share a pixel;
boundary F1 confirms these are single-pixel false positives at true
grain boundaries.

---

## 7. Design Considerations

### 7.1. Why Stereographic Projection over Direct $q_{xyz}$ Encoding?

Encoding $(q_x, q_y, q_z)$ directly and reconstructing
$q_w = \sqrt{1 - q_x^2 - q_y^2 - q_z^2}$ fails when neural-network
noise pushes $\|\mathbf{q}_{xyz}\|^2 > 1$. The square root becomes
imaginary, producing NaN or requiring hard clamping that destroys
gradient flow. Stereographic projection avoids this entirely: its
inverse is a rational function with no domain restrictions.

### 7.2. Why Continuous Unfolding over FZ Folding?

Standard fundamental-zone folding selects the symmetry equivalent with
the largest $q_w$ *independently* for each grain. Two adjacent grains
with a $2°$ physical misorientation can be mapped near opposite edges
of the FZ, producing a large RGB colour jump. Neural networks (VAEs,
diffusion models) interpret such boundaries as high-frequency features
and blur them, irrecoverably mixing orientations from different grains.

Continuous unfolding via BFS guarantees that the RGB colour difference
between adjacent grains is proportional to their true physical
misorientation angle. The resulting images are smooth, low-frequency
colour fields that compress and reconstruct cleanly through bottleneck
architectures.

### 7.3. Why a Class-Level Mean Instead of Per-Sample?

A per-sample mean quaternion $\bar{\mathbf{q}}_j$ requires saving a
separate metadata file for each training image and, critically, the
decoder at inference time must know $\bar{\mathbf{q}}_j$ — which is
unavailable for a *generated* image. Using a single class-level mean
$\bar{\mathbf{q}}_c$ (computed once from a representative subsample
and stored in `class_means.json`) enables decoding with no per-sample
metadata. It also anchors the BFS consistently across all samples in
a class, ensuring that the same physical orientation maps to the same
RGB colour across different microstructures, a desirable property for
latent-space interpolation.

### 7.4. Why 8-bit PNG over 16-bit TIFF for Training?

Pretrained vision encoders (ViT, ResNet, CLIP) expect 8-bit RGB input.
Using 16-bit TIFF would require either (a) discarding the pretrained
weights and training from scratch, or (b) a lossy conversion at data
loading time, negating the precision advantage. Direct 8-bit PNG
encoding introduces only $\sim 0.6°$ mean error — well below the
$5°$–$15°$ misorientation threshold for grain boundaries (Read &
Shockley, 1950) and below the typical angular resolution of EBSD
detectors ($\sim 0.5°$–$1.0°$). File sizes are $\sim 13\times$
smaller, reducing I/O bottlenecks during training.

### 7.5. Image Resolution

The output image resolution is determined entirely by the spatial
discretisation of the input Dream3D simulation. For the synthetic RVE
database used in this work, all simulations are $300 \times 300$ voxels
in the $XY$-plane with $Z = 1$ (2D slices), yielding
$300 \times 300 \times 3$ RGB images. No resampling is performed
by the codec; any resolution adjustment for specific network
architectures (e.g., $224 \times 224$ for ViT-B) should be handled by
the data loader at training time.

---

## 8. References

1. Markley, F. L., Cheng, Y., Crassidis, J. L., & Oshman, Y. (2007).
   Averaging quaternions. *Journal of Guidance, Control, and Dynamics*,
   30(4), 1193–1197.

2. Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019). On the
   continuity of rotation representations in neural networks. In
   *Proceedings of the IEEE/CVF Conference on Computer Vision and
   Pattern Recognition* (CVPR), pp. 5745–5753.

3. Read, W. T., & Shockley, W. (1950). Dislocation models of crystal
   grain boundaries. *Physical Review*, 78(3), 275–289.

4. Groeber, M. A., & Jackson, M. A. (2014). DREAM.3D: A digital
   representation environment for the analysis of microstructure in 3D.
   *Integrating Materials and Manufacturing Innovation*, 3(1), 56–72.

5. Bunge, H. J. (1982). *Texture Analysis in Materials Science:
   Mathematical Methods*. Butterworths, London.
