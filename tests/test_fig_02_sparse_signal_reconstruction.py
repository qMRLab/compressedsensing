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

    # --- single delta at t = 0 ---

    def test_delta_at_t0_real_part_is_one_everywhere(self):
        """x[0]=1, all else 0 → X[k] = 1 (real=1, imag=0) for all k."""
        signal = make_sparse_signal(self.N, [0], [1.0])
        kspace = compute_kspace(signal)
        np.testing.assert_allclose(kspace.real, np.ones(self.N), atol=1e-10)
        np.testing.assert_allclose(kspace.imag, np.zeros(self.N), atol=1e-10)

    def test_delta_at_t0_magnitude_is_flat(self):
        """Delta at t=0 has flat magnitude spectrum equal to the spike height."""
        signal = make_sparse_signal(self.N, [0], [1.0])
        np.testing.assert_allclose(np.abs(compute_kspace(signal)), 1.0, atol=1e-10)

    # --- single delta at t = 1 ---

    def test_delta_at_t1_has_flat_magnitude(self):
        """x[1]=A → X[k] = A·exp(-j·2π·k/N), so |X[k]| = A for all k."""
        signal = make_sparse_signal(self.N, [1], [1.0])
        np.testing.assert_allclose(np.abs(compute_kspace(signal)), 1.0, atol=1e-10)

    def test_delta_at_t1_complex_values_match_linear_phase(self):
        """X[k] = exp(-j·2π·k/N): complex values encode a linear phase shift."""
        signal = make_sparse_signal(self.N, [1], [1.0])
        kspace = compute_kspace(signal)
        k = np.arange(self.N)
        expected = np.exp(-1j * 2 * np.pi * k / self.N)
        np.testing.assert_allclose(kspace, expected, atol=1e-10)

    # --- single delta at t = N//2 (middle) ---

    def test_delta_at_middle_has_flat_magnitude(self):
        """Delta anywhere in time → flat magnitude spectrum."""
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
        """X[0] = sum of all x[n], so it equals the total signal energy."""
        s = make_sparse_signal(self.N, [0, 1], [1.0, 1.0])
        kspace = compute_kspace(s)
        assert kspace[0].real == pytest.approx(2.0)
        assert kspace[0].imag == pytest.approx(0.0)

    def test_two_spikes_magnitude_follows_cosine_pattern(self):
        """x[0]=x[1]=1 → |X[k]| = 2|cos(π·k/N)|  (constructive/destructive interference)."""
        s = make_sparse_signal(self.N, [0, 1], [1.0, 1.0])
        kspace = compute_kspace(s)
        k = np.arange(self.N)
        expected_magnitude = 2 * np.abs(np.cos(np.pi * k / self.N))
        np.testing.assert_allclose(np.abs(kspace), expected_magnitude, atol=1e-10)
