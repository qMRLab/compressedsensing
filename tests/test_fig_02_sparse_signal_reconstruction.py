import numpy as np
import pytest

from figures.fig_02_sparse_signal_reconstruction import make_sparse_signal, compute_kspace


class TestMakeSparseSignal:
    def test_output_length(self):
        out = make_sparse_signal(length=256, positions=[10, 50], heights=[1.0, 2.0])
        assert len(out) == 256

    def test_output_is_ndarray(self):
        out = make_sparse_signal(length=64, positions=[5], heights=[3.0])
        assert isinstance(out, np.ndarray)

    def test_nonzero_values_at_correct_positions(self):
        out = make_sparse_signal(length=100, positions=[10, 40, 70], heights=[1.5, -2.0, 3.0])
        assert out[10] == 1.5
        assert out[40] == -2.0
        assert out[70] == 3.0

    def test_all_other_positions_are_zero(self):
        positions = [10, 40, 70]
        out = make_sparse_signal(length=100, positions=positions, heights=[1.5, -2.0, 3.0])
        zero_mask = np.ones(100, dtype=bool)
        zero_mask[positions] = False
        assert np.all(out[zero_mask] == 0.0)

    def test_empty_positions_gives_all_zero(self):
        out = make_sparse_signal(length=50, positions=[], heights=[])
        assert np.all(out == 0.0)
        assert len(out) == 50

    def test_mismatched_positions_heights_raises(self):
        with pytest.raises(ValueError):
            make_sparse_signal(length=100, positions=[10, 20], heights=[1.0])

    def test_out_of_bounds_position_raises(self):
        with pytest.raises((ValueError, IndexError)):
            make_sparse_signal(length=50, positions=[99], heights=[1.0])

    def test_negative_position_raises(self):
        with pytest.raises((ValueError, IndexError)):
            make_sparse_signal(length=50, positions=[-1], heights=[1.0])

    def test_sparsity(self):
        """Signal should have exactly as many nonzero elements as positions."""
        positions = [5, 20, 80, 120]
        out = make_sparse_signal(length=256, positions=positions, heights=[1.0, 2.0, -1.5, 0.5])
        assert np.count_nonzero(out) == len(positions)


class TestComputeKspace:
    """Tests for compute_kspace, verifying the fftshift-centered DFT convention.

    KEY CONVENTIONS (apply throughout this class and TestComputeKspaceOddN):

    * compute_kspace returns np.fft.fftshift(np.fft.fft(signal)).
    * DC component (k=0) is therefore at array index N//2, NOT at index 0.
    * The k-axis after fftshift runs from -N//2 (index 0) to N//2-1 (index N-1)
      for even N, or from -(N-1)//2 to +(N-1)//2 for odd N.
    * For even N the k-axis is ASYMMETRIC: there is a Nyquist bin at k=-N/2
      with no positive counterpart.  For odd N the k-axis is SYMMETRIC.

    DC / FLAT-SPECTRUM DUALITY:

        signal[0] = A, all else 0  →  X[k] = A  for ALL k  (FLAT spectrum)
        signal[n] = A for all n    →  X[k=0] = N·A, X[k≠0] = 0  (DC SPIKE)

    The first fact holds because exp(-j·2π·k·0/N) = 1 for every k, so
    the DFT sum collapses to x[0] regardless of k.  The second follows from
    the orthogonality of the DFT basis: the constant signal projects entirely
    onto the k=0 basis vector.

    More generally: any single spike at t=t₀ gives a FLAT magnitude spectrum
    (|X[k]| = |A| for all k) with a LINEAR PHASE RAMP (∠X[k] = -2π·k·t₀/N).
    The t₀=0 case is just the degenerate case where the phase slope is zero.
    """

    # Use N=8 throughout: small enough for exact manual verification.
    N = 8

    # --- output shape / type ---

    def test_output_length_matches_input(self):
        signal = make_sparse_signal(self.N, [0], [1.0])
        assert len(compute_kspace(signal)) == self.N

    def test_output_is_complex(self):
        signal = make_sparse_signal(self.N, [0], [1.0])
        assert np.iscomplexobj(compute_kspace(signal))

    # --- all-zeros input ---

    def test_all_zeros_gives_zero_kspace(self):
        """DFT of the zero signal is zero."""
        signal = make_sparse_signal(self.N, [], [])
        np.testing.assert_allclose(np.abs(compute_kspace(signal)), 0.0, atol=1e-10)

    # --- single delta at t = 0 (DC/flat duality, first direction) ---

    def test_delta_at_t0_real_part_is_one_everywhere(self):
        """x[0]=1, all else 0 → X[k] = 1 for all k (real=1, imag=0).

        This is the first direction of the DC/flat duality:
          spike at t=0  →  flat k-space (NOT a DC spike).
        Reason: exp(-j·2π·k·0/N) = 1 for every k, so X[k] = x[0] always.
        """
        signal = make_sparse_signal(self.N, [0], [1.0])
        kspace = compute_kspace(signal)
        np.testing.assert_allclose(kspace.real, np.ones(self.N), atol=1e-10)
        np.testing.assert_allclose(kspace.imag, np.zeros(self.N), atol=1e-10)

    def test_delta_at_t0_magnitude_is_flat(self):
        """x[0]=A → |X[k]| = A for all k.

        Flat magnitude spectrum is the hallmark of a spike at t=0.
        Contrast with all-ones signal which gives a spike at k=0 (DC).
        """
        signal = make_sparse_signal(self.N, [0], [1.0])
        np.testing.assert_allclose(np.abs(compute_kspace(signal)), 1.0, atol=1e-10)

    # --- single delta at t = 1 ---

    def test_delta_at_t1_has_flat_magnitude(self):
        """x[1]=A → X[k] = A·exp(-j·2π·k/N), so |X[k]| = A for all k.

        Any spike at any single time t₀ produces a flat magnitude spectrum.
        The time location only affects the PHASE (linear ramp), not the magnitude.
        """
        signal = make_sparse_signal(self.N, [1], [1.0])
        np.testing.assert_allclose(np.abs(compute_kspace(signal)), 1.0, atol=1e-10)

    def test_delta_at_t1_complex_values_match_linear_phase(self):
        """X[k] = exp(-j·2π·k/N): complex values encode a linear phase ramp.

        The phase slope is -2π·t₀/N per frequency bin.  For t₀=1 that gives
        a slope of -2π/N.  After fftshift, index i maps to k = i - N//2,
        so the expected value at index i is exp(-j·2π·(i - N//2)/N).
        """
        signal = make_sparse_signal(self.N, [1], [1.0])
        kspace = compute_kspace(signal)
        k_centered = np.arange(self.N) - self.N // 2
        expected = np.exp(-1j * 2 * np.pi * k_centered / self.N)
        np.testing.assert_allclose(kspace, expected, atol=1e-10)

    # --- single delta at t = N//2 (middle) ---

    def test_delta_at_middle_has_flat_magnitude(self):
        """Delta anywhere in time → flat magnitude spectrum (regardless of t₀)."""
        signal = make_sparse_signal(self.N, [self.N // 2], [1.0])
        np.testing.assert_allclose(np.abs(compute_kspace(signal)), 1.0, atol=1e-10)

    # --- single delta at t = N-1 (last sample) ---

    def test_delta_at_end_has_flat_magnitude(self):
        """Delta at the last sample still gives flat magnitude."""
        signal = make_sparse_signal(self.N, [self.N - 1], [1.0])
        np.testing.assert_allclose(np.abs(compute_kspace(signal)), 1.0, atol=1e-10)

    # --- amplitude doubling ---

    def test_doubling_amplitude_at_t0_doubles_kspace_magnitude(self):
        """Scaling x by 2 scales every |X[k]| by 2 (linearity)."""
        s1 = make_sparse_signal(self.N, [0], [1.0])
        s2 = make_sparse_signal(self.N, [0], [2.0])
        np.testing.assert_allclose(
            np.abs(compute_kspace(s2)), 2 * np.abs(compute_kspace(s1)), atol=1e-10
        )

    def test_doubling_amplitude_at_t1_doubles_kspace_magnitude(self):
        """Same linearity check, but for a spike at t=1."""
        s1 = make_sparse_signal(self.N, [1], [1.0])
        s2 = make_sparse_signal(self.N, [1], [2.0])
        np.testing.assert_allclose(
            np.abs(compute_kspace(s2)), 2 * np.abs(compute_kspace(s1)), atol=1e-10
        )

    # --- two spikes: t=0 and t=1 ---

    def test_two_spikes_kspace_equals_sum_of_individual_kspaces(self):
        """DFT is linear: FFT(x0 + x1) = FFT(x0) + FFT(x1)."""
        s_both = make_sparse_signal(self.N, [0, 1], [1.0, 1.0])
        s0 = make_sparse_signal(self.N, [0], [1.0])
        s1 = make_sparse_signal(self.N, [1], [1.0])
        np.testing.assert_allclose(
            compute_kspace(s_both),
            compute_kspace(s0) + compute_kspace(s1),
            atol=1e-10,
        )

    def test_two_spikes_dc_component_is_sum_of_heights(self):
        """DC component (at index N//2 after fftshift) equals sum of all x[n].

        The DC bin X[k=0] = Σ x[n], so it accumulates all signal values.
        After fftshift, DC is at index N//2 (not index 0).
        """
        s = make_sparse_signal(self.N, [0, 1], [1.0, 1.0])
        kspace = compute_kspace(s)
        assert kspace[self.N // 2].real == pytest.approx(2.0)
        assert kspace[self.N // 2].imag == pytest.approx(0.0)

    def test_two_spikes_magnitude_follows_cosine_pattern(self):
        """x[0]=x[1]=1 → |X[k]| = 2|cos(π·k/N)| (constructive/destructive interference).

        Using centered k-axis: k = index - N//2.
        Derivation: X[k] = 1 + exp(-j2πk/N) = exp(-jπk/N)·2·cos(πk/N), so
        |X[k]| = 2|cos(πk/N)|.
        """
        s = make_sparse_signal(self.N, [0, 1], [1.0, 1.0])
        kspace = compute_kspace(s)
        k_centered = np.arange(self.N) - self.N // 2
        expected_magnitude = 2 * np.abs(np.cos(np.pi * k_centered / self.N))
        np.testing.assert_allclose(np.abs(kspace), expected_magnitude, atol=1e-10)

    # --- all-ones signal: DC/flat duality, second direction ---

    def test_all_ones_dc_component_equals_N(self):
        """x[n]=1 for all n → X[k=0] = N, all other bins = 0.

        This is the second direction of the DC/flat duality:
          constant signal  →  DC spike in k-space (NOT a flat spectrum).
        The DC bin is at array index N//2 after fftshift.
        Contrast with a spike at t=0, which gives a FLAT k-space spectrum.
        """
        signal = np.ones(self.N)
        kspace = compute_kspace(signal)
        assert kspace[self.N // 2].real == pytest.approx(float(self.N))
        assert kspace[self.N // 2].imag == pytest.approx(0.0)

    def test_all_ones_non_dc_components_are_zero(self):
        """x[n]=1 for all n: all X[k] = 0 except the DC bin at index N//2.

        The orthogonality of the DFT basis guarantees that a constant signal
        has zero projection onto every basis vector except k=0.
        """
        signal = np.ones(self.N)
        kspace = compute_kspace(signal)
        mask = np.ones(self.N, dtype=bool)
        mask[self.N // 2] = False
        np.testing.assert_allclose(np.abs(kspace[mask]), 0.0, atol=1e-10)

    # --- single sinusoid ---

    def test_sinusoid_energy_at_correct_frequency_bins(self):
        """x[n] = sin(2π·k₀·n/N) → energy only at k=+k₀ and k=−k₀.

        After fftshift those are at indices N//2+k₀ and N//2−k₀.
        All other bins should be zero (up to floating-point tolerance).
        Note: indices are always N//2 ± k₀, NOT k₀ and N−k₀ (the raw FFT
        convention); using raw indices only works by coincidence for certain
        even N values.
        """
        k0 = 2
        n = np.arange(self.N)
        signal = np.sin(2 * np.pi * k0 * n / self.N)
        kspace = compute_kspace(signal)
        mask = np.ones(self.N, dtype=bool)
        mask[self.N // 2 + k0] = False
        mask[self.N // 2 - k0] = False
        np.testing.assert_allclose(np.abs(kspace[mask]), 0.0, atol=1e-10)

    def test_sinusoid_magnitude_at_frequency_bins_equals_N_over_2(self):
        """x[n] = sin(2π·k₀·n/N) → |X[+k₀]| = |X[−k₀]| = N/2.

        After fftshift: +k₀ at index N//2+k₀, −k₀ at index N//2−k₀.
        """
        k0 = 2
        n = np.arange(self.N)
        signal = np.sin(2 * np.pi * k0 * n / self.N)
        kspace = compute_kspace(signal)
        assert abs(kspace[self.N // 2 + k0]) == pytest.approx(self.N / 2)
        assert abs(kspace[self.N // 2 - k0]) == pytest.approx(self.N / 2)

    def test_sinusoid_frequency_bins_are_conjugate_symmetric(self):
        """X[k₀] = −jN/2 and X[−k₀] = +jN/2 (Hermitian symmetry of real input).

        For a real signal: X[−k] = X[k]*.
        sin encodes into ±j·N/2 (not ±N/2) because it is a purely imaginary
        combination of complex exponentials: sin = (e^{jθ} − e^{−jθ})/(2j).
        After fftshift: k=+k₀ at index N//2+k₀, k=−k₀ at index N//2−k₀.
        """
        k0 = 2
        n = np.arange(self.N)
        signal = np.sin(2 * np.pi * k0 * n / self.N)
        kspace = compute_kspace(signal)
        assert kspace[self.N // 2 + k0] == pytest.approx(-1j * self.N / 2, abs=1e-10)
        assert kspace[self.N // 2 - k0] == pytest.approx(+1j * self.N / 2, abs=1e-10)


class TestComputeKspaceOddN(TestComputeKspace):
    """Identical suite as TestComputeKspace but with odd N=255.

    Verifies that all behaviours hold regardless of even/odd array length.

    For odd N the k-axis after fftshift is SYMMETRIC:
        k ∈ {-(N-1)//2, …, 0, …, +(N-1)//2}
    every negative-frequency bin has an exact positive-frequency mirror.

    For even N (N=8 in the parent class) the k-axis is ASYMMETRIC:
        k ∈ {-N//2, …, 0, …, N//2-1}
    there is an extra Nyquist bin at k=-N/2 with no positive counterpart.

    DC is always at index N//2 regardless of parity.
    """
    N = 255
