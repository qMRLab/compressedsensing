# Introduction to Wavelet Decomposition

### A Discovery-Based Exam — 12 Questions

---

**Philosophy.** This exam is modelled after a Fourier decomposition exam by Prof. Normand Beaudoin (Université de Moncton, 2008). That exam forced you to *discover* the properties of spectral decomposition by trying incomplete tools and watching them fail. This exam follows the same arc for wavelets.

You will start with a kernel that cannot possibly work, fix its deficiencies one by one, and arrive at a working wavelet transform not because you memorised the axioms, but because you derived each one from necessity. By the end, you will understand why wavelets have the properties they do, how the forward and inverse transforms work, and how compression falls out naturally.

**Tools.** You are encouraged to use Maple, MATLAB, Python, or any computational tool. Show all important steps and explain your reasoning at every stage. A correct answer without explanation will receive at most half marks.

**Signal design.** The signal you will build contains a loud high frequency and a quiet low frequency inside a smooth window. The 1:10 amplitude ratio means the slow oscillation is nearly invisible in the raw signal. As you progress through the questions, you will watch the wavelet transform *reveal* the hidden frequency, separate the two components into different scales, remove the dominant one to expose the quiet one, and compress by discarding near-zero coefficients.

---

## Question 1. Build your signal

A) Define a signal h(x) of the form:

$$h(x) = w(x,\,\sigma) \cdot \bigl[\sin(\omega_1 x) + 10\cos(\omega_2 x)\bigr]$$

where w(x, σ) is a smooth window function (e.g. a Gaussian) with a width parameter σ, ω₁ is a low frequency with a small amplitude, and ω₂ is a high frequency with a large amplitude. The 1:10 amplitude ratio is deliberate: the slow oscillation should be nearly invisible in the raw signal but clearly visible in the wavelet coefficients at the appropriate scale. The use of sin for one component and cos for the other ensures that h(x) is neither symmetric nor anti-symmetric — a requirement inherited from the Fourier exam. Choose ω₁ and ω₂ sufficiently far apart. Ensure that h(x) tends rapidly to zero for large |x|.

B) Plot h(x). The fast oscillation (ω₂) should dominate visually. Can you see the slow oscillation (ω₁) at all, or is it buried? This is deliberate — you will later use the wavelet transform to reveal it.

C) Choose σ wide enough that several full periods of ω₁ fit inside the window. Now try a σ that is too narrow — so narrow that fewer than one or two periods of ω₁ fit inside. Plot both cases. In the narrow case, can you distinguish the slow oscillation from a simple slope or offset? Keep the wider σ for all subsequent questions.

> *Your signal plays the role of h(x) in the Fourier exam. The kernels you will try in the following questions play the role of cos, sin, and complex exponentials. The window width matters practically: if σ is too small, the low frequency doesn't have room to express itself, and no transform can recover what isn't there.*

---

## Question 2. Try a positive kernel

Using your signal from Question 1 (with the wider σ):

A) Choose a smooth, positive kernel φ(x) (for example a Gaussian) with a width parameter. Your kernel should be 32 samples long (with N = 1024, this ensures the kernel fits comfortably at every level of a 4-level decomposition, where the coarsest array has N/8 = 128 samples). Convolve it with your signal h(x) at three different kernel widths. Plot the convolution output in each case.

B) Can you detect the oscillatory structure of h(x) in the convolution output? Can you distinguish the two frequencies? Explain what the convolution is actually computing and why the output behaves the way it does.

C) What is the fundamental limitation of using a kernel with nonzero mean? What property must a kernel have to detect changes rather than just smooth?

> *This is the analog of trying cosines only in the Fourier exam. A positive kernel is structurally incapable of seeing the detail, just as cosines alone cannot capture the odd part of h(x).*

---

## Question 3. Try a rough, zero-mean kernel

A) Construct a zero-mean kernel: a function whose integral is zero. Make it deliberately rough — a sawtooth with positive and negative lobes, a clipped ramp, a triangular wave, or anything discontinuous. Convolve it with h(x) at several widths.

B) Compare the output with Question 2. Can you now detect changes in the signal? Can you see evidence of the oscillatory structure?

C) Examine the output carefully. Do you see spurious ringing, ripples, or artifacts that are not present in the original signal? If so, explain where they come from in terms of the kernel's frequency response. What property would the kernel need to eliminate these artifacts?

> *You have now discovered two of the core wavelet requirements: zero mean (∫ψ = 0) and smoothness. Each was motivated by a failure.*

---

## Question 4. The Haar wavelet at a single scale

The Haar wavelet is the simplest function that satisfies the zero-mean condition:

$$\psi(x) = \begin{cases} +1 & 0 \le x < \tfrac{1}{2} \\ -1 & \tfrac{1}{2} \le x < 1 \\ 0 & \text{otherwise} \end{cases}$$

A) Convolve h(x) with the Haar wavelet at a scale matched to the high frequency ω₂ (i.e., set the wavelet width to approximately one period of the fast oscillation). Plot the output. Which frequency component can you see in the coefficients?

B) Now set the scale to match the low frequency ω₁ (wavelet width ≈ one period of the slow oscillation). Plot the output. Which frequency component dominates?

C) Can a single, fixed-scale wavelet capture both frequency components simultaneously? Explain why or why not, and state what this implies about the requirements for a complete decomposition.

> *This is the wavelet analog of Questions 2–3 of the Fourier exam. Just as cosines alone could not capture the odd part, a wavelet at one scale cannot capture features at a different scale. The restriction is parity in Fourier, scale in wavelets.*

---

## Question 5. The filter bank: applying both filters

Instead of convolving with the wavelet at different widths directly, the discrete wavelet transform uses a pair of short convolution filters derived from the wavelet:

- **Low-pass filter (from the scaling function φ):** captures the smooth, slowly-varying part.
- **High-pass filter (from the wavelet ψ):** captures the fast-changing detail.

For the Haar wavelet, these are simply h = [1/√2, 1/√2] (low-pass) and g = [1/√2, −1/√2] (high-pass).

A) Discretise your signal h(x) into a vector of N samples (choose N to be a power of 2, e.g. 512 or 1024). Convolve with both filters and downsample by 2. You now have two output vectors of length N/2. Call them a₁ (from the low-pass) and d₁ (from the high-pass). Plot both.

B) What does a₁ look like compared to the original signal? What does d₁ look like? The original signal is dominated by the high-frequency component (ω₂). Can you now see the quiet low-frequency component (ω₁) emerging in a₁, even though it was nearly invisible in h(x)?

C) Explain why you need both a₁ and d₁ to represent the original signal. What information is in a₁ that is absent from d₁, and vice versa? How does this parallel the Fourier exam's finding that you need both cosines and sines?

> *The scaling function φ and the wavelet ψ are complementary, just as cos and sin are complementary. φ is "blind" to detail (like cos is blind to the odd part), and ψ is blind to the overall level (like sin is blind to the even part). Together they span the full space.*

---

## Question 6. The cascade: multi-scale decomposition

A) Take a₁ (the low-pass output from Question 5) and apply the same two filters again, followed by downsampling by 2. You now have a₂ (length N/4) and d₂ (length N/4). Repeat once more to get a₃ (length N/8) and d₃ (length N/8).

B) Plot d₁, d₂, d₃, and a₃ stacked vertically, aligned by position in the signal. At which level do the high-frequency oscillations (ω₂) appear? At which level do the low-frequency oscillations (ω₁) appear? What does a₃ contain?

C) Verify that the total number of coefficients stored (|d₁| + |d₂| + |d₃| + |a₃|) equals N, the original number of samples. Explain why this is the case and why it matters for compression.

D) Why does each level operate on the previous level's low-pass output rather than on the original signal? What would happen (in terms of redundancy and total number of coefficients) if every level operated on the original instead?

> *This is the central structural question. The cascade gives you a non-redundant decomposition. Operating on the original at every scale (the CWT approach) is valid but redundant — you get more coefficients than input samples and cannot compress.*

---

## Question 7. Reconstruction from partial information

> **How reconstruction works.** The decomposition at each level was: convolve with a filter, then downsample by 2. Reconstruction reverses both operations at each level, working from the coarsest level back to the finest.
>
> To reconstruct a_{k-1} from a_k and d_k:
> 1. **Upsample** a_k by 2: insert a zero after every sample, e.g. [x, y, z] → [x, 0, y, 0, z, 0].
> 2. **Convolve** the upsampled a_k with the synthesis low-pass filter h̃.
> 3. **Upsample** d_k by 2 in the same way.
> 4. **Convolve** the upsampled d_k with the synthesis high-pass filter g̃.
> 5. **Add** the two results element-wise: a_{k-1} = (h̃ * (↑2 a_k)) + (g̃ * (↑2 d_k)).
>
> For the Haar wavelet the synthesis filters are h̃ = [1/√2, 1/√2] and g̃ = [-1/√2, 1/√2] (the time-reversed analysis filters). For a 3-level decomposition you repeat this three times: a₃ + d₃ → a₂, then a₂ + d₂ → a₁, then a₁ + d₁ → h(t).

Using the checkboxes, reconstruct the signal keeping only one set of coefficients at a time. Try each in turn: keep only d₃ (128 coefficients), then only d₂ (256 coefficients), then only d₁ (512 coefficients), then only a₃ (128 coefficients). Each reconstruction isolates the content of one level. Compare each to the original signal (1024 samples). Note that storing all four sets of coefficients (d₁ + d₂ + d₃ + a₃) also requires 1024 values — the decomposition is non-redundant. At which level does the high-frequency component (ω₂) live? At which level does the low-frequency component (ω₁) appear? Which combination of levels would you keep to reveal the hidden slow oscillation while discarding the dominant fast one?

> *Each detail level captures a different frequency band. By selectively keeping or removing levels, you can isolate or suppress specific frequency components — and the memory cost of each choice is visible in the coefficient count.*

---

## Question 8. Computing the inverse

A) Write down the analysis (decomposition) filter coefficients for the Haar wavelet: h (low-pass) and g (high-pass). Now write down the synthesis (reconstruction) filters. How are they related to the analysis filters?

B) Reconstruct the signal using all coefficients. Compare the reconstruction to the original sample by sample. Is the reconstruction exact? Verify numerically.

C) Why does time-reversal of the filter produce the inverse? Express the full set of decomposition and reconstruction operations as a matrix applied to your signal vector. What property does this matrix have? What is its inverse?

D) Is a Fourier transform needed anywhere in this process? Where does Fourier analysis enter in the design of wavelet filters, and where does it not?

> *The key insight: for orthogonal wavelets, the transform matrix is orthogonal, so its inverse is its transpose. In convolution terms, transpose = time-reversal. The inverse is not computed — it is a structural consequence of orthogonality.*

---

## Question 9. Distance, thresholding, and compression

A) Define a distance measure between two signals (e.g. the L² norm of their difference). This must produce a single real number. Explain your choice.

B) Sort all your wavelet coefficients (d₁, d₂, d₃, a₃) by absolute value. Progressively set the smallest coefficients to zero, reconstructing after each step. Plot the reconstruction error (your distance measure) as a function of the percentage of coefficients zeroed out.

C) How many coefficients can you discard before the reconstruction error becomes perceptible? What is special about your signal that makes wavelet compression effective (or not effective)? Would a signal with only one frequency compress better or worse? Explain.

D) Now repeat the compression experiment using the signal with the narrow window (from Question 1C, where fewer than two periods of ω₁ fit). Can you still recover the low-frequency component after thresholding? What does this tell you about the relationship between the window width and the compressibility of different frequency components?

> *This is the wavelet analog of Question 6 from the Fourier exam. The distance measure is the same; the insight is that natural signals produce sparse wavelet coefficients, enabling compression by thresholding.*

---

## Question 10. A smoother wavelet

A) Replace the Haar wavelet with the Daubechies-4 wavelet (filter coefficients: h = [0.4830, 0.8365, 0.2241, −0.1294]). Derive the high-pass analysis filter g from h. Derive the synthesis filters. Repeat the full decomposition and reconstruction of your signal.

B) Compare the wavelet coefficients to those obtained with Haar. Which wavelet produces sparser coefficients (more near-zero values)? Which gives a better reconstruction when the same percentage of coefficients are thresholded?

C) The Daubechies-4 wavelet has two vanishing moments (∫ψ = 0 and ∫xψ = 0), while Haar has only one (∫ψ = 0). Explain what vanishing moments mean physically and why more vanishing moments lead to sparser coefficients for smooth signals.

D) The Daubechies-4 filters are not symmetric. What practical consequence does this have? If symmetry matters (e.g. for image compression), what type of wavelet would you use instead, and what property of orthogonal wavelets would you have to give up?

> *This parallels the Fourier exam's progression from crude tools to refined ones. Haar is the "sawtooth" of wavelets: it works, but its roughness wastes coefficients. Smoother wavelets achieve better compression for the same reason that smooth kernels produce fewer artifacts in Question 3.*

---

## Question 11. Properties of the coefficients

A) You have observed that the wavelet coefficients at the finest scale (d₁) are large only where the fast oscillation (ω₂) is present, and nearly zero in smooth regions. Explain why, in terms of what the high-pass filter responds to.

B) At the coarsest scales, the detail coefficients (d₃) are large only where the slow oscillation (ω₁) is present. But the coefficients become small at scales much coarser than 1/ω₁. Explain why wavelet coefficients tend to zero at both very fine and very coarse scales, and identify the scales at which they are maximal.

C) In the Fourier exam, the coefficients H(k) peaked at the signal's dominant frequencies. In your wavelet decomposition, the coefficients peak at specific scales. What is the relationship between the scale at which a coefficient is maximal and the local frequency content of the signal at that position?

> *This is the wavelet analog of Questions 11–12 of the Fourier exam. The Fourier coefficients are large at the signal's frequencies; the wavelet coefficients are large at the signal's frequencies AND positions. This dual localisation is the essential advantage.*

---

## Question 12. The big picture

Look back at everything you have done. Answer the following, point by point, clearly and systematically:

A) List every property of a wavelet that you discovered by encountering a failure (zero mean, smoothness, multi-scale, need for both φ and ψ, etc.). For each, state which question revealed it and what broke without it.

B) The Fourier exam concluded with spectral decomposition: any signal equals a sum of orthogonal basis functions weighted by coefficients. State the wavelet equivalent. What are the basis functions? What are the coefficients? In what sense is this decomposition "better" than Fourier for your signal, and in what sense is Fourier "better"?

C) Your signal contained two known frequencies inside a window. A real-world signal (audio, an image scanline, a seismic trace) has unknown, time-varying frequency content. Explain why the cascade structure of the DWT is particularly well-suited to such signals, and why a fixed-resolution Fourier transform is not.

D) If you were to extend this exam to two dimensions (images), how many detail sub-images would you expect per scale, and why? What spatial features would each one capture?

> *This is the analog of Question 13 of the Fourier exam. The goal is not to recite definitions but to demonstrate, from your own work, that you understand why wavelets have the structure they do.*

---

## Summary of the Pedagogical Arc

**Fourier exam:** cos alone fails → sin alone fails → cos + sin work (orthogonal complements) → complex exponentials unify → uncertainty product → spectral decomposition.

**This exam:** positive kernel fails (Q2) → rough zero-mean kernel has artifacts (Q3) → single-scale wavelet is incomplete (Q4) → filter bank with φ + ψ at one level (Q5) → multi-scale cascade separates frequencies (Q6) → need both φ and ψ for reconstruction (Q7) → inverse from orthogonality (Q8) → compression from sparsity (Q9) → smoother wavelets, better compression (Q10) → coefficient behaviour (Q11) → the big picture (Q12).

**The signal design** (sin + cos with a 1:10 amplitude ratio, in a window) is chosen so that the slow oscillation is nearly invisible in the raw signal. The wavelet transform reveals it at the coarse scale; zeroing d₁ erases the dominant fast oscillation and exposes the hidden slow one. Every abstract property has a visible, often dramatic, consequence in your plots.

---

*Bon travail.*