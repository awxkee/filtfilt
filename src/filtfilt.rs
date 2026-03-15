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
use crate::filtfilt_error::FiltfiltError;
use crate::mla::fmla;
use crate::pad::FilterPadding;
use crate::traits::{FilterSample, Filtering};
use num_traits::AsPrimitive;

/// Compute the initial conditions for [`lfilter_with_zi`] that correspond to
/// the steady state of the step response.
///
/// The returned vector `zi` has length `max(b.len(), a.len()) - 1`.  When the
/// filter order is zero (both `b` and `a` have length 1) there are no delay
/// elements and an empty vector is returned.
///
/// # Steady-state meaning
///
/// `zi` satisfies `(I - A) * zi = B` where
///
/// ```text
/// A = companion(a).T          (the state-transition matrix)
/// B = b[1:] - a[1:] * b[0]   (the input vector, after normalising a[0] = 1)
pub(crate) fn lfilter_zi_impl<T: FilterSample>(b: &[T], a: &[T]) -> Result<Vec<T>, FiltfiltError>
where
    f64: AsPrimitive<T>,
{
    if b.is_empty() {
        return Err(FiltfiltError::EmptyNumerator);
    }
    if a.is_empty() {
        return Err(FiltfiltError::EmptyDenominator);
    }
    if a[0] == T::zero() || !a[0].is_finite() {
        return Err(FiltfiltError::DenominatorLeadingZero);
    }
    if !b.iter().chain(a.iter()).all(|v| v.is_finite()) {
        return Err(FiltfiltError::NonFiniteCoefficients);
    }

    // ... compute denom ...

    let n = b.len().max(a.len());

    // Normalise so a[0] == 1
    let a0 = a[0];
    let rcp_a0 = T::one() / a0;

    let mut b_pad = vec![T::zero(); n];
    let mut a_pad = vec![T::zero(); n];
    for (dst, &v) in b_pad.iter_mut().zip(b.iter()) {
        *dst = v * rcp_a0;
    }
    for (dst, &v) in a_pad.iter_mut().zip(a.iter()) {
        *dst = v * rcp_a0;
    }

    let m = n - 1;
    if m == 0 {
        return Ok(vec![]);
    }

    // B = b[1:] - a[1:] * b[0]
    // zi[0] = B.sum() / (1 + a[1] + a[2] + ... + a[m])
    // zi[k] = asum * zi[0] - csum   for k in 1..m
    //
    // This is the explicit solution to (I - companion(a).T) * zi = B
    // from scipy's lfilter_zi, avoiding a full matrix solve.

    let b0 = b_pad[0];
    let b_sum: T = b_pad[1..].iter().sum::<T>();
    let a_tail_sum: T = a_pad[1..].iter().sum::<T>();

    // B.sum() = sum(b[1:]) - b[0] * sum(a[1:])
    let b_rhs_sum = fmla(-b0, a_tail_sum, b_sum);
    // IminusA[:,0].sum() = 1 + sum(a[1:])
    let denom = T::one() + a_tail_sum;

    // denom = 1 + sum(a[1:]) == 0 means a pole sits exactly at z=1.
    // zi[0] would be NaN or inf — catch it before the divide.
    if denom == T::zero() {
        return Err(FiltfiltError::UnstableAtDc);
    }

    let mut zi = vec![T::zero(); m];
    let z0 = b_rhs_sum / denom;
    zi[0] = z0;

    let mut asum = T::one();
    let mut csum = T::zero();

    for ((&a_pad, &b_pad), zi) in a_pad[1..]
        .iter()
        .zip(b_pad[1..].iter())
        .zip(zi[1..].iter_mut())
    {
        asum += a_pad;
        csum = fmla(-a_pad, b0, csum + b_pad);
        *zi = fmla(asum, z0, -csum);
    }

    Ok(zi)
}

pub struct LFilterState<T> {
    pub y: Vec<T>,
    pub zi: Vec<T>,
}

/// Builder for a single-direction IIR filter pass, constructed from
/// numerator (`b`) and denominator (`a`) coefficients.
///
/// Obtain via [`LFilterBuilder::new`], optionally set initial conditions
/// with [`.zi()`][LFilterBuilder::zi], then call
/// [`.filter()`][LFilterBuilder::filter] to run the filter.
pub struct LFilterBuilder<'a, T> {
    /// Numerator (FIR part) coefficients `[b0, b1, …, bP]`.
    pub b: &'a [T],
    /// Denominator (IIR part) coefficients `[a0, a1, …, aQ]`.
    /// `a[0]` is the leading term and must be non-zero; coefficients are
    /// normalised by `a[0]` internally.
    pub a: &'a [T],
    /// Optional initial filter state of length `max(b.len(), a.len()) - 1`.
    /// `None` (the default) starts the filter from all zeros.
    /// Set via [`.zi()`][LFilterBuilder::zi].
    pub zi: Option<&'a [T]>,
}

impl<'a, T: Filtering> LFilterBuilder<'a, T> {
    pub fn new(b: &'a [T], a: &'a [T]) -> Self {
        Self { b, a, zi: None }
    }

    /// Set initial conditions.  Length must equal `max(b.len(), a.len()) - 1`.
    /// If not called, the filter starts from all zeros.
    pub fn zi(mut self, zi: &'a [T]) -> Self {
        self.zi = Some(zi);
        self
    }

    pub fn filter(self, x: &[T]) -> Result<LFilterState<T>, FiltfiltError> {
        T::lfilter_with_zi(x, self)
    }
}

/// Apply an IIR filter to a signal with given initial conditions, returning
/// the filtered output and the final filter state.
///
/// Implements the Direct Form II transposed difference equation:
///
/// ```text
/// y[n] = b[0]*x[n] + z[0][n]
/// z[k][n+1] = b[k+1]*x[n] - a[k+1]*y[n] + z[k+1][n]   for k in 0..m-1
/// z[m-1][n+1] = b[m]*x[n] - a[m]*y[n]
/// ```
///
/// where `m = max(b.len(), a.len()) - 1` is the filter order.
///
/// # Arguments
///
/// - `b` — numerator coefficients, length ≥ 1.
/// - `a` — denominator coefficients, length ≥ 1.  `a[0]` is the leading
///   term; all coefficients are normalised by `a[0]` internally so you do
///   not need to normalise beforehand.
/// - `x` — input signal.  An empty slice is valid and produces an empty
///   output with `zf` equal to the input `zi` (no samples processed).
/// - `zi` — initial filter state, length must equal `max(b.len(), a.len()) - 1`.
///   Obtain a steady-state value from [`lfilter_zi`] and scale by `x[0]`:
///   `let zi: Vec<f64> = lfilter_zi(b, a)?.iter().map(|&z| z * x[0]).collect();`
pub(crate) fn lfilter_with_zi_impl<T: FilterSample>(
    x: &[T],
    options: LFilterBuilder<'_, T>,
) -> Result<LFilterState<T>, FiltfiltError> {
    let a = options.a;
    let b = options.b;
    let zi = options.zi;
    // 1. b and a must be non-empty
    if b.is_empty() {
        return Err(FiltfiltError::EmptyNumerator);
    }
    if a.is_empty() {
        return Err(FiltfiltError::EmptyDenominator);
    }

    // 2. a[0] must be finite and non-zero — it's the divisor for everything
    if a[0] == T::zero() || !a[0].is_finite() {
        return Err(FiltfiltError::DenominatorLeadingZero);
    }

    // 3. all coefficients must be finite — NaN/inf propagates silently otherwise
    if !b.iter().chain(a.iter()).all(|v| v.is_finite()) {
        return Err(FiltfiltError::NonFiniteCoefficients);
    }
    let n = x.len();
    let m = b.len().max(a.len()) - 1;
    let mut y = vec![T::zero(); n];
    // Validate or default zi
    let mut z: Vec<T> = match zi {
        Some(zi) => {
            if zi.len() != m {
                return Err(FiltfiltError::ZiLengthMismatch {
                    expected: m,
                    got: zi.len(),
                });
            }
            zi.to_vec()
        }
        None => vec![T::zero(); m],
    };
    z.resize(m.max(1), T::zero()); // ensure at least 1 element to avoid index panic

    let a0 = a[0];

    let mut b_full = vec![T::zero(); m + 1];
    let mut a_full = vec![T::zero(); m + 1];

    let rcp_a0 = T::one() / a0;

    for (&v, v_dst) in b.iter().zip(b_full.iter_mut()) {
        *v_dst = v * rcp_a0;
    }
    for (&v, v_dst) in a.iter().zip(a_full.iter_mut()) {
        *v_dst = v * rcp_a0;
    }

    for (&xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi = fmla(b_full[0], xi, z[0]);
        for ((j, &b_full), &a_full) in (0..m.saturating_sub(1))
            .zip(b_full[1..].iter())
            .zip(a_full[1..].iter())
        {
            z[j] = fmla(b_full, xi, fmla(-a_full, *yi, z[j + 1]));
        }
        if m > 0 {
            z[m - 1] = fmla(b_full[m], xi, -a_full[m] * *yi);
        }
    }

    Ok(LFilterState { y, zi: z })
}

pub(crate) fn filtfilt_impl<T: FilterSample + Filtering>(
    b: &[T],
    a: &[T],
    x: &[T],
    filter_padding: FilterPadding,
) -> Result<Vec<T>, FiltfiltError>
where
    f64: AsPrimitive<T>,
{
    if b.is_empty() {
        return Err(FiltfiltError::EmptyNumerator);
    }
    if a.is_empty() {
        return Err(FiltfiltError::EmptyDenominator);
    }
    if a[0] == T::zero() || !a[0].is_finite() {
        return Err(FiltfiltError::DenominatorLeadingZero);
    }
    if !b.iter().all(|v| v.is_finite()) || !a.iter().all(|v| v.is_finite()) {
        return Err(FiltfiltError::NonFiniteCoefficients);
    }
    if x.is_empty() {
        return Err(FiltfiltError::EmptySignal);
    }

    let mut a = std::borrow::Cow::Borrowed(a);
    let mut b = std::borrow::Cow::Borrowed(b);
    if (a[0] - T::one()).abs() > T::epsilon() {
        let a0 = a[0];
        let rcp_a0 = T::one() / a0;
        b = std::borrow::Cow::Owned(b.iter().map(|&v| v * rcp_a0).collect::<Vec<_>>());
        a = std::borrow::Cow::Owned(a.iter().map(|&v| v * rcp_a0).collect::<Vec<_>>());
    };

    let n = x.len();
    let pad = 3 * (b.len().max(a.len()));

    // Edge padding (reflect)
    let padded = filter_padding.extend(x, pad);

    // Initial conditions scaled to first sample
    let zi_base = T::lfilter_zi(b.as_ref(), a.as_ref())?;
    let zi: Vec<T> = zi_base.iter().map(|&z| z * padded[0]).collect();

    // Forward pass
    let LFilterState { y: forward, zi: _ } =
        T::lfilter_with_zi(&padded, LFilterBuilder::new(b.as_ref(), a.as_ref()).zi(&zi))?;

    // Reverse + backward pass
    let rev: Vec<T> = forward.into_iter().rev().collect();
    let zi_back: Vec<T> = zi_base.iter().map(|&z| z * rev[0]).collect();
    let LFilterState { y: backward, zi: _ } = T::lfilter_with_zi(
        &rev,
        LFilterBuilder::new(b.as_ref(), a.as_ref()).zi(&zi_back),
    )?;

    // Trim padding and reverse
    Ok(backward[pad..pad + n]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn assert_vec_approx(actual: &[f64], expected: &[f64], tol: f64, msg: &str) {
        assert_eq!(actual.len(), expected.len(), "{}: length mismatch", msg);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{}: index {} — got {}, expected {} (diff {})",
                msg,
                i,
                a,
                e,
                (a - e).abs()
            );
        }
    }

    fn no_nans(v: &[f64], msg: &str) {
        for (i, x) in v.iter().enumerate() {
            assert!(!x.is_nan(), "{}: NaN at index {}", msg, i);
            assert!(!x.is_infinite(), "{}: Inf at index {}", msg, i);
        }
    }

    // ------------------------------------------------------------------
    // lfilter_zi
    // ------------------------------------------------------------------

    #[test]
    fn test_lfilter_zi_identity_filter() {
        // b=[1], a=[1] → no state needed
        let zi = f64::lfilter_zi(&[1.0], &[1.0]).expect("Linear filter state got successfully");
        assert!(zi.is_empty());
    }

    #[test]
    fn test_lfilter_zi_simple_lowpass() {
        // Simple first-order: b=[0.5, 0.5], a=[1.0, -0.0]
        // scipy: lfilter_zi([0.5,0.5],[1.0,0.0]) == [0.5]
        let zi = f64::lfilter_zi(&[0.5, 0.5], &[1.0, 0.0])
            .expect("Linear filter state got successfully");
        assert_eq!(zi.len(), 1);
        assert!((zi[0] - 0.5).abs() < 1e-10, "zi[0] = {}", zi[0]);
    }

    #[test]
    fn test_lfilter_zi_length() {
        // zi length should be max(len(b), len(a)) - 1
        let b = vec![1.0, 2.0, 3.0];
        let a = vec![1.0, 0.5, 0.25];
        let zi = f64::lfilter_zi(&b, &a).expect("Linear filter state got successfully");
        assert_eq!(zi.len(), 2);
    }

    #[test]
    fn test_lfilter_zi_no_nans() {
        let b = vec![0.1, 0.2, 0.1];
        let a = vec![1.0, -0.5, 0.1];
        let zi = f64::lfilter_zi(&b, &a).expect("Linear filter state got successfully");
        no_nans(&zi, "lfilter_zi");
    }

    // ------------------------------------------------------------------
    // lfilter_with_zi
    // ------------------------------------------------------------------

    #[test]
    fn test_lfilter_with_zi_passthrough() {
        // b=[1], a=[1], zi=[] → output == input
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let LFilterState { y, zi: _ } =
            f64::lfilter_with_zi(&x, LFilterBuilder::new(&[1.0], &[1.0]))
                .expect("Linear filter executed successfully");
        assert_vec_approx(&y, &x, TOL, "passthrough");
    }

    #[test]
    fn test_lfilter_with_zi_constant_signal_settled() {
        // With correct zi a constant signal should pass through unchanged
        let b = vec![0.5, 0.5];
        let a = vec![1.0, 0.0];
        let x = vec![3.0; 20];
        let zi_base = f64::lfilter_zi(&b, &a).expect("Linear filter state got successfully");
        let zi: Vec<f64> = zi_base.iter().map(|&z| z * x[0]).collect();
        let LFilterState { y, zi: _ } =
            f64::lfilter_with_zi(&x, LFilterBuilder::new(&b, &a).zi(&zi))
                .expect("Linear filter executed successfully");
        // All outputs should equal 3.0 (settled)
        for (i, &v) in y.iter().enumerate() {
            assert!((v - 3.0).abs() < 1e-10, "index {}: got {}", i, v);
        }
    }

    #[test]
    fn test_lfilter_with_zi_output_length() {
        let x = vec![1.0; 100];
        let b = vec![0.1, 0.2, 0.1];
        let a = vec![1.0, -0.5, 0.1];
        let zi = vec![0.0; 2];
        let LFilterState { y, zi: zf } =
            f64::lfilter_with_zi(&x, LFilterBuilder::new(&b, &a).zi(&zi))
                .expect("Linear filter executed successfully");
        assert_eq!(y.len(), 100);
        assert_eq!(zf.len(), 2);
    }

    #[test]
    fn test_lfilter_with_zi_no_nans() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let b = vec![0.2, 0.5, 0.2];
        let a = vec![1.0, -0.3, 0.1];
        let zi = vec![0.0; 2];
        let LFilterState { y, zi: _ } =
            f64::lfilter_with_zi(&x, LFilterBuilder::new(&b, &a).zi(&zi))
                .expect("Linear filter executed successfully");
        no_nans(&y, "lfilter_with_zi");
    }

    // ------------------------------------------------------------------
    // filtfilt_impl
    // ------------------------------------------------------------------

    #[test]
    fn test_filtfilt_impl_output_length() {
        let x: Vec<f64> = (0..200).map(|i| (i as f64).sin()).collect();
        let b = vec![0.1, 0.2, 0.1];
        let a = vec![1.0, -0.5, 0.1];
        let y = filtfilt_impl(&b, &a, &x, FilterPadding::Odd).unwrap();
        assert_eq!(y.len(), x.len());
    }

    #[test]
    fn test_filtfilt_impl_no_nans() {
        let x: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let b = vec![
            4.3909323578773363e-07,
            0.0,
            -1.7563729431509345e-06,
            0.0,
            2.6345594147264016e-06,
            0.0,
            -1.7563729431509345e-06,
            0.0,
            4.3909323578773363e-07,
        ];
        let a = vec![
            1.0,
            -7.852836991432248,
            26.990584149386436,
            -53.03237713543542,
            65.15251936216197,
            -51.248690406703766,
            25.205541117878923,
            -7.086848740900965,
            0.8721086450898761,
        ];
        let y = filtfilt_impl(&b, &a, &x, FilterPadding::Odd).unwrap();
        no_nans(&y, "filtfilt_impl bandpass");
    }

    #[test]
    fn test_filtfilt_impl_zero_phase() {
        // filtfilt_impl must have zero phase: applying to a pure sine at passband
        // frequency should preserve phase (output peak aligns with input peak)
        let fs = 100.0f64;
        let n = 512;
        let freq = 0.3; // Hz — inside [0.15, 0.5] band
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();
        let b = vec![0.1, 0.2, 0.1];
        let a = vec![1.0, -0.5, 0.1];
        let y = filtfilt_impl(&b, &a, &x, FilterPadding::Odd).unwrap();

        // Find peak index in middle portion (avoid edges)
        let peak_x = x[50..n - 50]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i + 50)
            .unwrap();
        let peak_y = y[50..n - 50]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i + 50)
            .unwrap();
        assert!(
            (peak_x as i64 - peak_y as i64).abs() <= 2,
            "phase shift detected: x peak {}, y peak {}",
            peak_x,
            peak_y
        );
    }

    #[test]
    fn test_filtfilt_impl_constant_signal() {
        // A constant signal through any stable filter should remain constant
        let x = vec![5.0f64; 100];
        let b = vec![0.5, 0.5];
        let a = vec![1.0, 0.0];
        let y = filtfilt_impl(&b, &a, &x, FilterPadding::Odd).unwrap();
        for (i, &v) in y.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-8, "index {}: got {}", i, v);
        }
    }

    #[test]
    fn test_filtfilt_impl_attenuates_high_freq() {
        // Low-pass filter should reduce high-frequency amplitude
        let fs = 100.0f64;
        let n = 512;
        // High frequency signal (well above cutoff)
        let x_high: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 40.0 * i as f64 / fs).sin())
            .collect();
        // Low-pass like coefficients
        let b = vec![0.5, 0.5];
        let a = vec![1.0, 0.0];
        let y = filtfilt_impl(&b, &a, &x_high, FilterPadding::Odd).unwrap();

        let rms_in: f64 = (x_high.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let rms_out: f64 = (y.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        assert!(
            rms_out < rms_in * 0.5,
            "expected attenuation, rms_in={} rms_out={}",
            rms_in,
            rms_out
        );
    }

    #[test]
    fn test_filtfilt_impl_matches_scipy_known_output() {
        // Pre-computed with scipy.signal.filtfilt_impl on x = sin(0.2 * arange(10))
        // b = [0.2, 0.5, 0.2], a = [1.0, -0.3, 0.05]
        let x: Vec<f64> = (0..10).map(|i| (i as f64 * 0.2).sin()).collect();
        let b = vec![0.2, 0.5, 0.2];
        let a = vec![1.0, -0.3, 0.05];
        let expected = [
            -0.0, 0.2787264, 0.5463412, 0.7921743, 1.0064125, 1.1804632, 1.3073317, 1.3826565,
            1.4087276, 1.4023399,
        ];
        let y = filtfilt_impl(&b, &a, &x, FilterPadding::Odd).unwrap();
        println!("{:#?}", y);
        for (i, (a, e)) in y.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-3,
                "index {}: got {}, expected {}",
                i,
                a,
                e
            );
        }
    }
}
