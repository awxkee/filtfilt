/*
 * // Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::FilterPadding;
use crate::filtfilt_error::FiltfiltError;
use crate::mla::fmla;
use crate::traits::{FilterSample, Filtering};
use num_traits::AsPrimitive;

#[derive(Debug, Copy, Clone)]
pub struct SosFilter<T> {
    pub b: [T; 3],
    pub a: [T; 3],
}

impl<T> SosFilter<T> {
    pub fn new(b: [T; 3], a: [T; 3]) -> Self {
        Self { b, a }
    }
}

impl<T: Copy> From<[T; 6]> for SosFilter<T> {
    fn from(row: [T; 6]) -> Self {
        Self {
            b: [row[0], row[1], row[2]],
            a: [row[3], row[4], row[5]],
        }
    }
}

/// Compute the initial conditions for one biquad section (b, a already
/// normalised so a[0] == 1).  Returns `[z0, z1]`.
///
/// Solves `(I - companion(a).T) * zi = b[1:] - a[1:]*b[0]` using the
/// explicit recurrence from scipy's `lfilter_zi`:
///
/// ```text
/// zi[0] = (b[1] + b[2] - (a[1] + a[2])*b[0]) / (1 + a[1] + a[2])
/// zi[1] = a[2]*zi[0] - (b[2] - a[2]*b[0])
/// ```
pub(crate) fn sosfilt_zi_impl<T: FilterSample + Filtering>(
    sos: &[SosFilter<T>],
) -> Result<Vec<[T; 2]>, FiltfiltError> {
    if sos.is_empty() {
        return Err(FiltfiltError::EmptyNumerator);
    }
    let mut zi = vec![[T::zero(); 2]; sos.len()];
    let mut scale = T::one();

    for (dst, s) in zi.iter_mut().zip(sos.iter()) {
        let [z0, z1] = T::lfilter_zi(&s.b, &s.a)?.try_into().unwrap();
        *dst = [scale * z0, scale * z1];

        // DC gain of this section: sum(b) / sum(a)
        let sum_b = s.b[0] + s.b[1] + s.b[2];
        let sum_a = s.a[0] + s.a[1] + s.a[2];
        scale *= sum_b / sum_a;
    }

    Ok(zi)
}

/// Run one biquad section (Direct Form II transposed) over `x` in place,
/// using and updating `z[0..2]`.
fn sosfilt_section_inplace<T: FilterSample>(b: &[T; 3], a: &[T; 3], x: &mut [T], z: &mut [T; 2]) {
    let rcp_a0 = T::one() / a[0];
    let b0 = b[0] * rcp_a0;
    let b1 = b[1] * rcp_a0;
    let b2 = b[2] * rcp_a0;
    let a1 = a[1] * rcp_a0;
    let a2 = a[2] * rcp_a0;

    for xi in x.iter_mut() {
        let x_in = *xi;
        let y = fmla(b0, x_in, z[0]);
        z[0] = fmla(b1, x_in, fmla(-a1, y, z[1]));
        z[1] = fmla(b2, x_in, -a2 * y);
        *xi = y;
    }
}

/// Apply a digital filter forward and backward over a signal using
/// second-order sections (SOS), matching scipy's `sosfiltfilt`.
///
/// The algorithm: for each `SosFilter` section run `filtfilt_impl` with that
/// section's `(b, a)`. The output of each section feeds into the next.
pub(crate) fn sosfiltfilt_impl<T: FilterSample + Filtering>(
    sos: &[SosFilter<T>],
    x: &[T],
    padding: FilterPadding,
) -> Result<Vec<T>, FiltfiltError>
where
    f64: AsPrimitive<T>,
{
    if sos.is_empty() {
        return Err(FiltfiltError::EmptyNumerator);
    }
    if x.is_empty() {
        return Err(FiltfiltError::EmptySignal);
    }
    for s in sos {
        if s.a[0] == T::zero() || !s.a[0].is_finite() {
            return Err(FiltfiltError::DenominatorLeadingZero);
        }
        if !s.b.iter().chain(s.a.iter()).all(|v| v.is_finite()) {
            return Err(FiltfiltError::NonFiniteCoefficients);
        }
    }

    let n_sections = sos.len();

    // Count trailing zeros in b[:,2] and a[:,2] to match scipy's ntaps formula
    let trailing_b_zeros = sos.iter().filter(|s| s.b[2] == T::zero()).count();
    let trailing_a_zeros = sos.iter().filter(|s| s.a[2] == T::zero()).count();
    let ntaps = 2 * n_sections + 1 - trailing_b_zeros.min(trailing_a_zeros);
    let edge = ntaps * 3;

    if x.len() <= edge {
        return Err(FiltfiltError::SignalTooShort {
            signal_len: x.len(),
            required: edge,
        });
    }

    // Pad signal
    let mut ext = padding.extend(x, edge);

    // sosfilt_zi: one [z0, z1] per section
    let zi: Vec<[T; 2]> = T::sosfilt_zi(sos)?;

    // ── forward pass ─────────────────────────────────────────────────────────
    let x0 = ext[0];
    let mut z_fwd: Vec<[T; 2]> = zi.iter().map(|&[z0, z1]| [z0 * x0, z1 * x0]).collect();

    for (s, z) in sos.iter().zip(z_fwd.iter_mut()) {
        sosfilt_section_inplace(&s.b, &s.a, &mut ext, z);
    }

    // ── backward pass ────────────────────────────────────────────────────────
    let y0 = ext[ext.len() - 1];
    let mut z_back: Vec<[T; 2]> = zi.iter().map(|&[z0, z1]| [z0 * y0, z1 * y0]).collect();

    ext.reverse();
    for (s, z) in sos.iter().zip(z_back.iter_mut()) {
        sosfilt_section_inplace(&s.b, &s.a, &mut ext, z);
    }
    ext.reverse();

    // Strip padding
    Ok(ext[edge..ext.len() - edge].to_vec())
}

/// Output of a single-direction SOS filter pass.
pub struct SosFilterState<T> {
    /// Filtered signal, same length as input.
    pub y: Vec<T>,
    /// Final state for each section, shape `(n_sections, 2)`.
    /// Pass back as `zi` to continue filtering the next chunk.
    pub zf: Vec<[T; 2]>,
}

/// Builder for a single-direction SOS filter pass.
///
/// Obtain via [`SosFilterBuilder::new`], optionally set per-section initial
/// conditions with [`.zi()`][SosFilterBuilder::zi], then call
/// [`.filter()`][SosFilterBuilder::filter] to run the filter.
pub struct SosFilterBuilder<'a, T> {
    /// Second-order sections to apply in cascade order.
    pub sos: &'a [SosFilter<T>],
    /// Optional per-section initial state, shape `(n_sections, 2)`.
    /// `None` (the default) starts every section from all zeros.
    /// Set via [`.zi()`][SosFilterBuilder::zi].
    pub zi: Option<&'a [[T; 2]]>,
}

impl<'a, T: Filtering> SosFilterBuilder<'a, T> {
    /// Create a new builder for the given SOS filter.
    /// Initial conditions default to all zeros.
    pub fn new(sos: &'a [SosFilter<T>]) -> Self {
        Self { sos, zi: None }
    }

    /// Set per-section initial conditions.
    ///
    /// `zi` must have length equal to `sos.len()`, with each element
    /// `[z0, z1]` being the two delay-line values for that section.
    /// Use [`sosfilt_zi`] to compute the steady-state conditions for a
    /// step response, then scale by `x[0]`:
    pub fn zi(mut self, zi: &'a [[T; 2]]) -> Self {
        self.zi = Some(zi);
        self
    }

    /// Run the filter over `x`, consuming the builder.
    ///
    /// Returns [`SosFilterState`] containing the filtered signal `y` and
    /// the final per-section state `zf`.  Pass `zf` back via
    /// [`.zi()`][SosFilterBuilder::zi] on the next call to continue
    /// filtering in chunks without discontinuities.
    pub fn filter(self, x: &[T]) -> Result<SosFilterState<T>, FiltfiltError> {
        T::sosfilt(x, self)
    }
}

/// Apply a cascade of second-order sections to a signal in one direction,
/// matching `scipy.signal.sosfilt`.
///
/// Each section is a Direct Form II transposed biquad:
///
/// ```text
/// y[n]     = b0*x[n] + z[0][n]
/// z[0][n+1] = b1*x[n] - a1*y[n] + z[1][n]
/// z[1][n+1] = b2*x[n] - a2*y[n]
/// ```
///
/// Sections are applied in order; the output of section `k` is the input
/// to section `k+1`.
///
/// # Arguments
///
/// - `sos` — second-order sections.
/// - `x`   — input signal.
/// - `zi`  — initial state, one `[z0, z1]` per section.  Pass `None` to
///   start from all zeros.  Use [`sosfilt_zi`] scaled by `x[0]` to start
///   from steady state.
///
pub(crate) fn sosfilt_impl<T: FilterSample + Filtering>(
    x: &[T],
    options: SosFilterBuilder<'_, T>,
) -> Result<SosFilterState<T>, FiltfiltError> {
    let sos = options.sos;
    let zi = options.zi;
    if sos.is_empty() {
        return Err(FiltfiltError::EmptyNumerator);
    }
    if x.is_empty() {
        return Err(FiltfiltError::EmptySignal);
    }
    for s in sos {
        if s.a[0] == T::zero() || !s.a[0].is_finite() {
            return Err(FiltfiltError::DenominatorLeadingZero);
        }
        if !s.b.iter().chain(s.a.iter()).all(|v| v.is_finite()) {
            return Err(FiltfiltError::NonFiniteCoefficients);
        }
    }
    if let Some(zi) = zi
        && zi.len() != sos.len()
    {
        return Err(FiltfiltError::ZiLengthMismatch {
            expected: sos.len(),
            got: zi.len(),
        });
    }

    let mut buf = x.to_vec();
    let mut zf: Vec<[T; 2]> = match zi {
        Some(zi) => zi.to_vec(),
        None => vec![[T::zero(), T::zero()]; sos.len()],
    };

    for (s, z) in sos.iter().zip(zf.iter_mut()) {
        sosfilt_section_inplace(&s.b, &s.a, &mut buf, z);
    }

    Ok(SosFilterState { y: buf, zf })
}

#[cfg(test)]
mod sosfilt_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // How to reproduce these values in Python:
    //
    //   import numpy as np
    //   from scipy.signal import butter, sosfilt, sosfilt_zi
    //   rng = np.random.default_rng(42)
    //   x = rng.standard_normal(64)
    //
    //   # test 1
    //   sos = butter(2, 0.1, output='sos')
    //   y = sosfilt(sos, x)
    //   print(y[:8].tolist())
    //
    //   # test 2
    //   sos = butter(4, [0.1, 0.3], btype='bandpass', output='sos')
    //   zi = sosfilt_zi(sos) * x[0]
    //   y, zf = sosfilt(sos, x, zi=zi)
    //   print(y[:8].tolist())
    //   print(zf.flatten().tolist())
    //
    //   # test 3
    //   sos = butter(2, 0.1, output='sos')
    //   y_first, zf = sosfilt(sos, x[:32])
    //   y_second, _ = sosfilt(sos, x[32:], zi=zf)
    //   assert np.allclose(sosfilt(sos, x)[0], np.concatenate([y_first, y_second]))
    // -----------------------------------------------------------------------

    const ATOL: f64 = 1e-10;
    const RTOL: f64 = 1e-10;

    fn assert_close(got: &[f64], expected: &[f64], label: &str) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
            let err = (g - e).abs();
            assert!(
                err <= ATOL + RTOL * e.abs(),
                "{label}[{i}]: got {g:.17e}, expected {e:.17e}, err={err:.3e}",
            );
        }
    }

    fn x64() -> Vec<f64> {
        vec![
            3.04717079754431353e-01,
            -1.03998410624049553e+00,
            7.50451195806457250e-01,
            9.40564716391213862e-01,
            -1.95103518865383641e+00,
            -1.30217950686231809e+00,
            1.27840403167285371e-01,
            -3.16242592343582207e-01,
            -1.68011575042887953e-02,
            -8.53043927573580052e-01,
            8.79397974862828558e-01,
            7.77791935428948311e-01,
            6.60306975612160452e-02,
            1.12724120696803287e+00,
            4.67509342252045601e-01,
            -8.59292462883238239e-01,
            3.68750784082498839e-01,
            -9.58882600828998899e-01,
            8.78450301307272530e-01,
            -4.99259109862528958e-02,
            -1.84862363545260561e-01,
            -6.80929544403941378e-01,
            1.22254133867403025e+00,
            -1.54529482068802154e-01,
            -4.28327822163107219e-01,
            -3.52133550488229585e-01,
            5.32309185553348718e-01,
            3.65444064364078336e-01,
            4.12732611595988397e-01,
            4.30821003007882730e-01,
            2.14164760087046124e+00,
            -4.06415016384615579e-01,
            -5.12242729071537339e-01,
            -8.13772728247877719e-01,
            6.15979422575495650e-01,
            1.12897229272089161e+00,
            -1.13947457654875073e-01,
            -8.40156476962528043e-01,
            -8.24481215691239555e-01,
            6.50592787824701091e-01,
            7.43254171203442282e-01,
            5.43154268305194976e-01,
            -6.65509707288694297e-01,
            2.32161323066719771e-01,
            1.16685809140728222e-01,
            2.18688596729012946e-01,
            8.71428777948189848e-01,
            2.23595548774682268e-01,
            6.78913563071894877e-01,
            6.75790694888914606e-02,
            2.89119398689984153e-01,
            6.31288225838540384e-01,
            -1.45715581985566645e+00,
            -3.19671216357301335e-01,
            -4.70372654292795511e-01,
            -6.38877848243341928e-01,
            -2.75142251226683732e-01,
            1.49494131123439589e+00,
            -8.65831115693243225e-01,
            9.68278354591480817e-01,
            -1.68286977161580475e+00,
            -3.34885029985774851e-01,
            1.62753065105005590e-01,
            5.86222331359278148e-01,
        ]
    }

    // -----------------------------------------------------------------------
    // Test 1: butter(2, 0.1) lowpass — zero initial conditions
    //
    // Python: y = sosfilt(butter(2, 0.1, output='sos'), x)
    // -----------------------------------------------------------------------
    #[test]
    fn test_lowpass_order2_zero_zi() {
        let sos = vec![SosFilter::new(
            [
                2.00833655642112321e-02,
                4.01667311284224643e-02,
                2.00833655642112321e-02,
            ],
            [
                1.00000000000000000e+00,
                -1.56101807580071816e+00,
                6.41351538057563064e-01,
            ],
        )];
        let state = f64::sosfilt(&x64(), SosFilterBuilder::new(&sos)).unwrap();

        assert_eq!(state.y.len(), 64);
        assert_eq!(state.zf.len(), 1);

        let expected_y = vec![
            6.11974450636715381e-03,
            9.06139819858216289e-04,
            -2.30918386770234407e-02,
            -8.48143628902186739e-03,
            1.52379537494433462e-02,
            -5.64028444252086814e-02,
            -1.86738926253152582e-01,
            -2.82697219868040139e-01,
        ];
        assert_close(&state.y[..8], &expected_y, "lowpass order=2 zero zi, y[:8]");
    }

    // -----------------------------------------------------------------------
    // Test 2: butter(4, [0.1, 0.3]) bandpass — steady-state initial conditions
    //
    // Python:
    //   sos = butter(4, [0.1, 0.3], btype='bandpass', output='sos')
    //   zi = sosfilt_zi(sos) * x[0]
    //   y, zf = sosfilt(sos, x, zi=zi)
    // -----------------------------------------------------------------------
    #[test]
    fn test_bandpass_order4_steady_state_zi() {
        let sos = vec![
            SosFilter::new(
                [
                    4.82434335771623254e-03,
                    9.64868671543246507e-03,
                    4.82434335771623254e-03,
                ],
                [
                    1.00000000000000000e+00,
                    -1.10547167301169935e+00,
                    4.68726607533460515e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.48782202096284299e+00,
                    6.31797625315156264e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    -2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.04431444730426581e+00,
                    7.20629642532835346e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    -2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.78062324708942521e+00,
                    8.78036033407334604e-01,
                ],
            ),
        ];

        // sosfilt_zi(sos) * x[0], row-major
        let zi = vec![
            [1.47175778968445300e-02, -6.11751669115874735e-03],
            [4.33545156651283348e-01, -2.67952473791400081e-01],
            [-4.49732794367823652e-01, 4.49732794367823652e-01],
            [-0.00000000000000000e+00, 0.00000000000000000e+00],
        ];

        let state = f64::sosfilt(&x64(), SosFilterBuilder::new(&sos).zi(&zi)).unwrap();

        assert_eq!(state.y.len(), 64);
        assert_eq!(state.zf.len(), 4);

        let expected_y = vec![
            2.22044604925031308e-16,
            -6.48730023476783851e-03,
            -3.29993193357044287e-02,
            -6.20121876981686929e-02,
            -4.08379482970795890e-02,
            1.84554408943422565e-02,
            2.38109995170552313e-02,
            -3.59678305977591747e-02,
        ];
        assert_close(
            &state.y[..8],
            &expected_y,
            "bandpass order=4 steady-state zi, y[:8]",
        );

        // Final state zf (flat: sec0_z0, sec0_z1, sec1_z0, ...)
        let expected_zf_flat = vec![
            -4.98695177750088127e-03,
            1.22571718996599965e-02,
            -1.64293438789407525e-01,
            8.33152650508227344e-02,
            -2.01830961217186855e-01,
            1.04203598655855212e-01,
            -2.64177080145390708e-01,
            2.34877980257890862e-01,
        ];
        let got_zf_flat: Vec<f64> = state.zf.iter().flat_map(|&[a, b]| [a, b]).collect();
        assert_close(&got_zf_flat, &expected_zf_flat, "bandpass order=4 zf");
    }

    // -----------------------------------------------------------------------
    // Test 3: chunk continuity — two halves equal one full pass
    //
    // Python:
    //   sos = butter(2, 0.1, output='sos')
    //   y_first, zf = sosfilt(sos, x[:32])
    //   y_second, _ = sosfilt(sos, x[32:], zi=zf)
    //   assert np.allclose(sosfilt(sos, x)[0], np.concatenate([y_first, y_second]))
    // -----------------------------------------------------------------------
    #[test]
    fn test_chunk_continuity() {
        let sos = vec![SosFilter::new(
            [
                2.00833655642112321e-02,
                4.01667311284224643e-02,
                2.00833655642112321e-02,
            ],
            [
                1.00000000000000000e+00,
                -1.56101807580071816e+00,
                6.41351538057563064e-01,
            ],
        )];
        let x = x64();

        // Full pass
        let full = f64::sosfilt(&x, SosFilterBuilder::new(&sos)).unwrap();

        // Two-chunk pass: state from first chunk seeds second chunk
        let first = f64::sosfilt(&x[..32], SosFilterBuilder::new(&sos)).unwrap();
        let second = f64::sosfilt(&x[32..], SosFilterBuilder::new(&sos).zi(&first.zf)).unwrap();

        let chunked: Vec<f64> = first.y.iter().chain(&second.y).copied().collect();
        assert_close(&chunked, &full.y, "chunk continuity");

        // Also spot-check the intermediate zf against scipy
        let expected_zf = vec![4.00569068889392699e-01, -2.09407195272952124e-01];
        let got_zf: Vec<f64> = first.zf.iter().flat_map(|&[a, b]| [a, b]).collect();
        assert_close(&got_zf, &expected_zf, "zf after first chunk");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Tolerance
    // -----------------------------------------------------------------------

    const ATOL: f64 = 1e-10;
    const RTOL: f64 = 1e-10;

    // -----------------------------------------------------------------------
    // Test 1: butter(2, 0.1) lowpass – odd padding
    // Python: sosfiltfilt(butter(2, 0.1, output='sos'), x, padtype='odd')
    // -----------------------------------------------------------------------
    #[test]
    fn test_lowpass_order2_odd() {
        let sos = vec![SosFilter::new(
            [
                2.00833655642112321e-02,
                4.01667311284224643e-02,
                2.00833655642112321e-02,
            ],
            [
                1.00000000000000000e+00,
                -1.56101807580071816e+00,
                6.41351538057563064e-01,
            ],
        )];
        let y = f64::sosfiltfilt(&sos, &x64(), FilterPadding::Odd).unwrap();
        let expected = vec![
            2.74580936351050742e-01,
            1.35280552472216237e-01,
            1.10988572306797324e-02,
            -9.18128838204060466e-02,
            -1.65975830510862121e-01,
            -2.04075972728244320e-01,
            -2.05029912984381446e-01,
            -1.74995007298055089e-01,
        ];
        assert_close(&y[..8], &expected, "lowpass order=2 odd padding");
    }

    fn assert_close(got: &[f64], expected: &[f64], label: &str) {
        assert_eq!(
            got.len(),
            expected.len(),
            "{label}: length mismatch (got {}, want {})",
            got.len(),
            expected.len()
        );
        for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
            let err = (g - e).abs();
            let tol = ATOL + RTOL * e.abs();
            assert!(
                err <= tol,
                "{label}[{i}]: got {g:.17e}, expected {e:.17e}, abs_err={err:.3e}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Fixtures
    // -----------------------------------------------------------------------

    /// 64-sample white-noise signal, numpy seed=42.
    fn x64() -> Vec<f64> {
        vec![
            3.04717079754431353e-01,
            -1.03998410624049553e+00,
            7.50451195806457250e-01,
            9.40564716391213862e-01,
            -1.95103518865383641e+00,
            -1.30217950686231809e+00,
            1.27840403167285371e-01,
            -3.16242592343582207e-01,
            -1.68011575042887953e-02,
            -8.53043927573580052e-01,
            8.79397974862828558e-01,
            7.77791935428948311e-01,
            6.60306975612160452e-02,
            1.12724120696803287e+00,
            4.67509342252045601e-01,
            -8.59292462883238239e-01,
            3.68750784082498839e-01,
            -9.58882600828998899e-01,
            8.78450301307272530e-01,
            -4.99259109862528958e-02,
            -1.84862363545260561e-01,
            -6.80929544403941378e-01,
            1.22254133867403025e+00,
            -1.54529482068802154e-01,
            -4.28327822163107219e-01,
            -3.52133550488229585e-01,
            5.32309185553348718e-01,
            3.65444064364078336e-01,
            4.12732611595988397e-01,
            4.30821003007882730e-01,
            2.14164760087046124e+00,
            -4.06415016384615579e-01,
            -5.12242729071537339e-01,
            -8.13772728247877719e-01,
            6.15979422575495650e-01,
            1.12897229272089161e+00,
            -1.13947457654875073e-01,
            -8.40156476962528043e-01,
            -8.24481215691239555e-01,
            6.50592787824701091e-01,
            7.43254171203442282e-01,
            5.43154268305194976e-01,
            -6.65509707288694297e-01,
            2.32161323066719771e-01,
            1.16685809140728222e-01,
            2.18688596729012946e-01,
            8.71428777948189848e-01,
            2.23595548774682268e-01,
            6.78913563071894877e-01,
            6.75790694888914606e-02,
            2.89119398689984153e-01,
            6.31288225838540384e-01,
            -1.45715581985566645e+00,
            -3.19671216357301335e-01,
            -4.70372654292795511e-01,
            -6.38877848243341928e-01,
            -2.75142251226683732e-01,
            1.49494131123439589e+00,
            -8.65831115693243225e-01,
            9.68278354591480817e-01,
            -1.68286977161580475e+00,
            -3.34885029985774851e-01,
            1.62753065105005590e-01,
            5.86222331359278148e-01,
        ]
    }

    /// butter(2, 0.1, output='sos')
    fn sos2() -> Vec<SosFilter<f64>> {
        vec![SosFilter::new(
            [
                2.00833655642112321e-02,
                4.01667311284224643e-02,
                2.00833655642112321e-02,
            ],
            [
                1.00000000000000000e+00,
                -1.56101807580071816e+00,
                6.41351538057563064e-01,
            ],
        )]
    }

    /// butter(4, 0.2, output='sos')
    fn sos4() -> Vec<SosFilter<f64>> {
        vec![
            SosFilter::new(
                [
                    4.82434335771622820e-03,
                    9.64868671543245640e-03,
                    4.82434335771622820e-03,
                ],
                [
                    1.00000000000000000e+00,
                    -1.04859957636261170e+00,
                    2.96140357561669620e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.32091343081942636e+00,
                    6.32738792885276569e-01,
                ],
            ),
        ]
    }

    /// butter(6, 0.15, output='sos')
    fn sos6() -> Vec<SosFilter<f64>> {
        vec![
            SosFilter::new(
                [
                    7.62145490093100178e-05,
                    1.52429098018620036e-04,
                    7.62145490093100178e-05,
                ],
                [
                    1.00000000000000000e+00,
                    -1.23878126513851550e+00,
                    3.90316716554843735e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.34896774525279484e+00,
                    5.13981894219675883e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.59464056877718496e+00,
                    7.89706949934815161e-01,
                ],
            ),
        ]
    }

    // -----------------------------------------------------------------------
    // From<[f64; 6]> conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_row() {
        let row: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s = SosFilter::from(row);
        assert_eq!(s.b, [1.0, 2.0, 3.0]);
        assert_eq!(s.a, [4.0, 5.0, 6.0]);
    }

    // -----------------------------------------------------------------------
    // Golden-value tests — odd padding
    // -----------------------------------------------------------------------

    #[test]
    fn test_sosfiltfilt_order2_golden_first16() {
        let y = f64::sosfiltfilt(&sos2(), &x64(), FilterPadding::Odd).unwrap();
        let expected = vec![
            2.74580936351050742e-01,
            1.35280552472216237e-01,
            1.10988572306797324e-02,
            -9.18128838204060466e-02,
            -1.65975830510862121e-01,
            -2.04075972728244320e-01,
            -2.05029912984381446e-01,
            -1.74995007298055089e-01,
            -1.23047879789334166e-01,
            -5.88839223115195515e-02,
            6.60677557357199421e-03,
            6.23613056564584767e-02,
            1.00459206823561478e-01,
            1.17926976630049002e-01,
            1.16772523380369447e-01,
            1.03455220041093782e-01,
        ];
        assert_close(&y[..16], &expected, "sosfiltfilt order=2 first 16");
    }

    #[test]
    fn test_sosfiltfilt_order2_full() {
        let y = f64::sosfiltfilt(&sos2(), &x64(), FilterPadding::Odd).unwrap();
        let expected = vec![
            2.74580936351050742e-01,
            1.35280552472216237e-01,
            1.10988572306797324e-02,
            -9.18128838204060466e-02,
            -1.65975830510862121e-01,
            -2.04075972728244320e-01,
            -2.05029912984381446e-01,
            -1.74995007298055089e-01,
            -1.23047879789334166e-01,
            -5.88839223115195515e-02,
            6.60677557357199421e-03,
            6.23613056564584767e-02,
            1.00459206823561478e-01,
            1.17926976630049002e-01,
            1.16772523380369447e-01,
            1.03455220041093782e-01,
            8.60321896885239135e-02,
            7.05073492968773807e-02,
            5.96735275078796867e-02,
            5.45379018912644092e-02,
            5.63711525703122357e-02,
            6.63852378415058370e-02,
            8.39689614502029807e-02,
            1.07717440328092301e-01,
            1.37475653799763731e-01,
            1.72739581451437418e-01,
            2.09807014234298872e-01,
            2.42294002819823495e-01,
            2.63760773102203694e-01,
            2.69190721480148165e-01,
            2.56235974292209068e-01,
            2.27958083301488035e-01,
            1.92997391053458006e-01,
            1.60058582615049849e-01,
            1.32626081309664606e-01,
            1.10190763594129867e-01,
            9.36072542733087376e-02,
            8.70747360098083623e-02,
            9.39653672785392274e-02,
            1.12295860592460134e-01,
            1.35887708143433344e-01,
            1.59469782792933085e-01,
            1.81274141888239176e-01,
            2.00737401256607106e-01,
            2.15463888199873860e-01,
            2.21036422860962034e-01,
            2.12619764283714791e-01,
            1.86822407357523562e-01,
            1.43102008266460079e-01,
            8.40502719188039177e-02,
            1.49202195584849200e-02,
            -5.66023726162347834e-02,
            -1.20610251414531022e-01,
            -1.67379164137356956e-01,
            -1.90886439901081506e-01,
            -1.89602067659536883e-01,
            -1.65475626806872333e-01,
            -1.22210780335319527e-01,
            -6.16674656687080380e-02,
            1.84150759631664281e-02,
            1.22517750502795786e-01,
            2.52787088448675923e-01,
            4.04806277394578806e-01,
            5.67395470950488034e-01,
        ];
        assert_close(&y, &expected, "sosfiltfilt order=2 full");
    }

    #[test]
    fn test_sosfiltfilt_order4_golden_first16() {
        let y = f64::sosfiltfilt(&sos4(), &x64(), FilterPadding::Odd).unwrap();
        let expected = vec![
            2.98714814211247126e-01,
            3.30179733233647715e-02,
            -2.14634902975778918e-01,
            -4.22661229541562766e-01,
            -5.63308069372815856e-01,
            -6.10164437474112198e-01,
            -5.49902959804463798e-01,
            -3.90939244023999377e-01,
            -1.64354083823239377e-01,
            8.24774845296023101e-02,
            2.97008897672359662e-01,
            4.34996501676943925e-01,
            4.73033443772870821e-01,
            4.15517282858351655e-01,
            2.92351585282971282e-01,
            1.47309231677258956e-01,
        ];
        assert_close(&y[..16], &expected, "sosfiltfilt order=4 first 16");
    }

    #[test]
    fn test_sosfiltfilt_order4_full() {
        let y = f64::sosfiltfilt(&sos4(), &x64(), FilterPadding::Odd).unwrap();
        let expected = vec![
            2.98714814211247126e-01,
            3.30179733233647715e-02,
            -2.14634902975778918e-01,
            -4.22661229541562766e-01,
            -5.63308069372815856e-01,
            -6.10164437474112198e-01,
            -5.49902959804463798e-01,
            -3.90939244023999377e-01,
            -1.64354083823239377e-01,
            8.24774845296023101e-02,
            2.97008897672359662e-01,
            4.34996501676943925e-01,
            4.73033443772870821e-01,
            4.15517282858351655e-01,
            2.92351585282971282e-01,
            1.47309231677258956e-01,
            2.15481827301184196e-02,
            -6.02663673360870553e-02,
            -9.48748566714063685e-02,
            -9.51709964742139952e-02,
            -7.91352348773143727e-02,
            -5.91965555612132752e-02,
            -3.63380667956085743e-02,
            -6.69133167908483699e-04,
            6.15840203224392382e-02,
            1.57260592127071691e-01,
            2.77382636381663383e-01,
            3.96197047340372366e-01,
            4.79852918331631184e-01,
            5.00817143338378901e-01,
            4.50796761439961113e-01,
            3.45041353823670960e-01,
            2.14978029721545949e-01,
            9.34523964565403265e-02,
            2.25038933247509801e-03,
            -5.09681376898612482e-02,
            -6.79506283018126156e-02,
            -5.46341500039066175e-02,
            -1.95880065359597920e-02,
            2.65078079201089660e-02,
            7.43630300574619207e-02,
            1.21310455622185592e-01,
            1.72634596147995756e-01,
            2.35841335452270429e-01,
            3.11277691144428992e-01,
            3.86047897100347148e-01,
            4.35850502495085335e-01,
            4.34179020699026474e-01,
            3.64378564465514865e-01,
            2.29038747385706787e-01,
            5.25399654618709452e-02,
            -1.25051279089355538e-01,
            -2.61644782854796198e-01,
            -3.29700144490177260e-01,
            -3.28096338437130375e-01,
            -2.82202293456516262e-01,
            -2.30271357663661247e-01,
            -2.01730379466777354e-01,
            -1.98512932515916507e-01,
            -1.90666619075498145e-01,
            -1.30528404567323342e-01,
            2.11009110126181010e-02,
            2.71119554925285999e-01,
            5.86064791937429552e-01,
        ];
        assert_close(&y, &expected, "sosfiltfilt order=4 full");
    }

    #[test]
    fn test_sosfiltfilt_order6_golden_first16() {
        let y = f64::sosfiltfilt(&sos6(), &x64(), FilterPadding::Odd).unwrap();
        let expected = vec![
            2.98991756836905598e-01,
            2.88230151179659604e-02,
            -2.09734863067139288e-01,
            -3.90286161948560917e-01,
            -4.94637663952731066e-01,
            -5.15366974072593687e-01,
            -4.56753811261759568e-01,
            -3.33890745823714585e-01,
            -1.70092744342109720e-01,
            6.96776575263441389e-03,
            1.69730631783605701e-01,
            2.95034600009331516e-01,
            3.67472669963650689e-01,
            3.81262679795832338e-01,
            3.40388253308144650e-01,
            2.57115660589813444e-01,
        ];
        assert_close(&y[..16], &expected, "sosfiltfilt order=6 first 16");
    }

    // -----------------------------------------------------------------------
    // Structural / correctness properties
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_length_matches_input() {
        for &n in &[64, 128, 513, 1000] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
            let y = f64::sosfiltfilt(&sos4(), &x, FilterPadding::Odd).unwrap();
            assert_eq!(y.len(), n, "length mismatch for n={n}");
        }
    }

    #[test]
    fn test_all_finite() {
        let y = f64::sosfiltfilt(&sos6(), &x64(), FilterPadding::Odd).unwrap();
        assert!(y.iter().all(|v| v.is_finite()), "output must be finite");
    }

    #[test]
    fn test_zero_signal_gives_zero_output() {
        let x = vec![0.0f64; 200];
        let y = f64::sosfiltfilt(&sos4(), &x, FilterPadding::Odd).unwrap();
        for (i, &v) in y.iter().enumerate() {
            assert!(v.abs() < 1e-14, "y[{i}] = {v}, expected 0 for zero input");
        }
    }

    #[test]
    fn test_dc_preserved() {
        let dc = 3.7_f64;
        let x = vec![dc; 500];
        let y = f64::sosfiltfilt(&sos2(), &x, FilterPadding::Odd).unwrap();
        for &v in &y[50..450] {
            let err = (v - dc).abs();
            assert!(err < 1e-8, "DC not preserved: got {v:.17e}, expected {dc}");
        }
    }

    #[test]
    fn test_zero_phase() {
        let n = 2000usize;
        let freq = 5.0_f64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 1000.0).sin())
            .collect();
        let y = f64::sosfiltfilt(&sos4(), &x, FilterPadding::Odd).unwrap();

        let trim_x = &x[200..n - 200];
        let trim_y = &y[200..n - 200];
        let m = trim_x.len();
        let best_lag = (-5isize..=5)
            .max_by(|&la, &lb| {
                let corr = |lag: isize| -> f64 {
                    (0..m)
                        .filter_map(|i| {
                            let j = (i as isize + lag) as usize;
                            if j < m {
                                Some(trim_x[i] * trim_y[j])
                            } else {
                                None
                            }
                        })
                        .sum()
                };
                corr(la).partial_cmp(&corr(lb)).unwrap()
            })
            .unwrap();
        assert_eq!(
            best_lag, 0,
            "non-zero phase lag of {best_lag} samples detected"
        );
    }

    #[test]
    fn test_high_freq_attenuated() {
        let n = 2000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| (std::f64::consts::PI * 0.9 * i as f64).sin())
            .collect();
        let y = f64::sosfiltfilt(&sos2(), &x, FilterPadding::Odd).unwrap();

        let rms = |v: &[f64]| (v.iter().map(|s| s * s).sum::<f64>() / v.len() as f64).sqrt();
        assert!(
            rms(&y) < 0.05 * rms(&x),
            "stopband signal not attenuated: rms_in={:.4}, rms_out={:.4}",
            rms(&x),
            rms(&y)
        );
    }

    #[test]
    fn test_linearity() {
        let alpha = 2.3_f64;
        let beta = -1.7_f64;
        let n = 64;
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let combined: Vec<f64> = u
            .iter()
            .zip(&v)
            .map(|(&a, &b)| alpha * a + beta * b)
            .collect();

        let yu = f64::sosfiltfilt(&sos4(), &u, FilterPadding::Odd).unwrap();
        let yv = f64::sosfiltfilt(&sos4(), &v, FilterPadding::Odd).unwrap();
        let y_combined = f64::sosfiltfilt(&sos4(), &combined, FilterPadding::Odd).unwrap();
        let y_expected: Vec<f64> = yu
            .iter()
            .zip(&yv)
            .map(|(&a, &b)| alpha * a + beta * b)
            .collect();

        assert_close(
            &y_combined,
            &y_expected,
            "linearity: f(αu + βv) == αf(u) + βf(v)",
        );
    }

    #[test]
    fn test_a0_scaling_invariance() {
        let x = x64();
        let sos_scaled: Vec<SosFilter<f64>> = sos4()
            .iter()
            .map(|s| SosFilter::new(s.b.map(|v| v * 4.2), s.a.map(|v| v * 4.2)))
            .collect();
        let y1 = f64::sosfiltfilt(&sos4(), &x, FilterPadding::Odd).unwrap();
        let y2 = f64::sosfiltfilt(&sos_scaled, &x, FilterPadding::Odd).unwrap();
        assert_close(&y1, &y2, "a0-scaling invariance");
    }

    // -----------------------------------------------------------------------
    // Error / validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_empty_sos() {
        let err = f64::sosfiltfilt(&[], &x64(), FilterPadding::Odd);
        assert!(matches!(err, Err(FiltfiltError::EmptyNumerator)));
    }

    #[test]
    fn test_error_empty_signal() {
        let err = f64::sosfiltfilt(&sos2(), &[], FilterPadding::Odd);
        assert!(matches!(err, Err(FiltfiltError::EmptySignal)));
    }

    #[test]
    fn test_error_a0_zero() {
        let s = sos2()[0];
        let bad_sos = vec![SosFilter::new(s.b, [0.0, s.a[1], s.a[2]])];
        let err = f64::sosfiltfilt(&bad_sos, &x64(), FilterPadding::Odd);
        assert!(matches!(err, Err(FiltfiltError::DenominatorLeadingZero)));
    }

    #[test]
    fn test_error_non_finite_coeff() {
        let s = sos2()[0];
        let bad_sos = vec![SosFilter::new([s.b[0], f64::NAN, s.b[2]], s.a)];
        let err = f64::sosfiltfilt(&bad_sos, &x64(), FilterPadding::Odd);
        assert!(matches!(err, Err(FiltfiltError::NonFiniteCoefficients)));
    }

    #[test]
    fn test_short_signal() {
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let y = f64::sosfiltfilt(&sos2(), &x, FilterPadding::Odd).unwrap();
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // -----------------------------------------------------------------------
    // Even padding golden values
    // -----------------------------------------------------------------------

    #[test]
    fn test_sosfiltfilt_order2_even_padding_full() {
        let y = f64::sosfiltfilt(&sos2(), &x64(), FilterPadding::Even).unwrap();
        let expected = vec![
            -2.68258319927786482e-01,
            -2.60874159086016022e-01,
            -2.59155152231427433e-01,
            -2.59610036523228338e-01,
            -2.54582650905751273e-01,
            -2.34776183513949077e-01,
            -1.96125766090721465e-01,
            -1.41406288179251938e-01,
            -7.63264399282470168e-02,
            -7.49353961189733880e-03,
            5.68628546045155114e-02,
            1.07852457949492789e-01,
            1.39240007697802010e-01,
            1.49289242340480438e-01,
            1.40858650428706517e-01,
            1.20941892546174060e-01,
            9.78846132954192399e-02,
            7.77984391177296841e-02,
            6.34592476070511557e-02,
            5.57785875169398079e-02,
            5.58885855551845626e-02,
            6.48460279606783013e-02,
            8.18860483400004613e-02,
            1.05463000788547062e-01,
            1.35300188194899779e-01,
            1.70793338230018715e-01,
            2.08161124173866785e-01,
            2.40959704821116644e-01,
            2.62705919067690086e-01,
            2.68353364149154283e-01,
            2.55535293153757059e-01,
            2.27303216946290382e-01,
            1.92294944767643888e-01,
            1.59219687717679470e-01,
            1.31573910482174844e-01,
            1.08869358613437592e-01,
            9.19923947937243136e-02,
            8.51873577792099518e-02,
            9.18877059919579880e-02,
            1.10190029499442371e-01,
            1.34015788463132696e-01,
            1.58213694628614343e-01,
            1.81152558530514707e-01,
            2.02415778786264017e-01,
            2.19752340888848263e-01,
            2.28868759048569070e-01,
            2.25005636596955383e-01,
            2.04763421764347364e-01,
            1.67462054611984268e-01,
            1.15370649880660828e-01,
            5.31717974392861906e-02,
            -1.23341660485923668e-02,
            -7.25057240554060761e-02,
            -1.19318830631807163e-01,
            -1.48915539915911277e-01,
            -1.62383748418521601e-01,
            -1.64669680907561178e-01,
            -1.62688862262999140e-01,
            -1.61446380007760737e-01,
            -1.61328678116185786e-01,
            -1.59394085402875807e-01,
            -1.53114964619677552e-01,
            -1.43580552453192456e-01,
            -1.34465766599383169e-01,
        ];
        assert_close(&y, &expected, "sosfiltfilt order=2 even padding full");
    }

    #[test]
    fn test_sosfiltfilt_order4_even_padding_full() {
        let y = f64::sosfiltfilt(&sos4(), &x64(), FilterPadding::Even).unwrap();
        let expected = vec![
            -1.77660757051424055e-02,
            -6.52086985356567367e-02,
            -1.91736707418679792e-01,
            -3.56632082693305763e-01,
            -5.04131479937051208e-01,
            -5.80179804020618195e-01,
            -5.50439994021451273e-01,
            -4.11557039299094263e-01,
            -1.91430517422722768e-01,
            5.98437009166948169e-02,
            2.84388029693008748e-01,
            4.32772423331623191e-01,
            4.78171013154773006e-01,
            4.23768460718395135e-01,
            3.00034752853706177e-01,
            1.52257267254481327e-01,
            2.32332616561841086e-02,
            -6.11675012933864393e-02,
            -9.71330265016602384e-02,
            -9.75896858400170236e-02,
            -8.09097566108825106e-02,
            -6.00177442889614082e-02,
            -3.63022968706720600e-02,
            -9.51340673255533092e-05,
            6.23381295344186820e-02,
            1.57920855876118982e-01,
            2.77805371388596900e-01,
            3.96352335032728098e-01,
            4.79777321423480174e-01,
            5.00568207365150730e-01,
            4.50430590742996795e-01,
            3.44617680023474204e-01,
            2.14581151859260300e-01,
            9.32064273642827790e-02,
            2.30683536684250096e-03,
            -5.04821906721082542e-02,
            -6.70163594227823622e-02,
            -5.34268176808141565e-02,
            -1.85168705471598360e-02,
            2.68559979092532868e-02,
            7.34064408826155951e-02,
            1.18762744190327979e-01,
            1.68821191230145967e-01,
            2.31898365849276339e-01,
            3.09065930686959756e-01,
            3.87658153606740219e-01,
            4.42707337136272694e-01,
            4.45952236476886921e-01,
            3.78128571883184106e-01,
            2.39154217539232106e-01,
            5.19600325158159904e-02,
            -1.42185275317927573e-01,
            -2.96383946964041800e-01,
            -3.74953727338124876e-01,
            -3.67377701177236393e-01,
            -2.92306815893779104e-01,
            -1.88912564876526912e-01,
            -9.93864614971956273e-02,
            -5.13609632319289022e-02,
            -4.93592152566618009e-02,
            -7.89190774585054045e-02,
            -1.18821009691510993e-01,
            -1.52173202302356342e-01,
            -1.70010015206849469e-01,
        ];
        assert_close(&y, &expected, "sosfiltfilt order=4 even padding full");
    }

    // -----------------------------------------------------------------------
    // Padding variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_odd_and_even_padding_differ() {
        let x = x64();
        let y_odd = f64::sosfiltfilt(&sos4(), &x, FilterPadding::Odd).unwrap();
        let y_even = f64::sosfiltfilt(&sos4(), &x, FilterPadding::Even).unwrap();
        let max_diff = y_odd
            .iter()
            .zip(&y_even)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-6,
            "odd and even padding should produce different outputs"
        );
    }

    #[test]
    fn test_both_padding_modes_finite_and_correct_length() {
        let x = x64();
        for padding in [FilterPadding::Odd, FilterPadding::Even] {
            let y = f64::sosfiltfilt(&sos4(), &x, padding).unwrap();
            assert_eq!(y.len(), x.len(), "{padding:?}: length mismatch");
            assert!(
                y.iter().all(|v| v.is_finite()),
                "{padding:?}: non-finite output"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: butter(4, [0.1, 0.3]) bandpass – even padding
    // Python: sosfiltfilt(butter(4, [0.1, 0.3], btype='bandpass', output='sos'), x, padtype='even')
    // -----------------------------------------------------------------------
    #[test]
    fn test_bandpass_order4_even() {
        let sos = vec![
            SosFilter::new(
                [
                    4.82434335771623254e-03,
                    9.64868671543246507e-03,
                    4.82434335771623254e-03,
                ],
                [
                    1.00000000000000000e+00,
                    -1.10547167301169935e+00,
                    4.68726607533460515e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.48782202096284299e+00,
                    6.31797625315156264e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    -2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.04431444730426581e+00,
                    7.20629642532835346e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    -2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.78062324708942521e+00,
                    8.78036033407334604e-01,
                ],
            ),
        ];
        let y = f64::sosfiltfilt(&sos, &x64(), FilterPadding::Even).unwrap();
        let expected = vec![
            0.4515574309472776,
            0.38976720607858445,
            0.21902043983896896,
            -0.026290530557970926,
            -0.281523244291266,
            -0.4675397597485283,
            -0.5227393962892309,
            -0.42714975521560633,
        ];
        assert_close(&y[..8], &expected, "bandpass order=4 even padding");
    }

    // -----------------------------------------------------------------------
    // Test 3: butter(6, 0.15) lowpass – odd padding
    // Python: sosfiltfilt(butter(6, 0.15, output='sos'), x, padtype='odd')
    // -----------------------------------------------------------------------
    #[test]
    fn test_lowpass_order6_odd() {
        let sos = vec![
            SosFilter::new(
                [
                    7.62145490093100178e-05,
                    1.52429098018620036e-04,
                    7.62145490093100178e-05,
                ],
                [
                    1.00000000000000000e+00,
                    -1.23878126513851550e+00,
                    3.90316716554843735e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.34896774525279484e+00,
                    5.13981894219675883e-01,
                ],
            ),
            SosFilter::new(
                [
                    1.00000000000000000e+00,
                    2.00000000000000000e+00,
                    1.00000000000000000e+00,
                ],
                [
                    1.00000000000000000e+00,
                    -1.59464056877718496e+00,
                    7.89706949934815161e-01,
                ],
            ),
        ];
        let y = f64::sosfiltfilt(&sos, &x64(), FilterPadding::Odd).unwrap();
        let expected = vec![
            2.98991756836905598e-01,
            2.88230151179659604e-02,
            -2.09734863067139288e-01,
            -3.90286161948560917e-01,
            -4.94637663952731066e-01,
            -5.15366974072593687e-01,
            -4.56753811261759568e-01,
            -3.33890745823714585e-01,
        ];
        assert_close(&y[..8], &expected, "lowpass order=6 odd padding");
    }
}
