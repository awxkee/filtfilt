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

fn lfilter_zi(b: &[f64], a: &[f64]) -> Vec<f64> {
    let n = b.len().max(a.len());
    let mut b_pad = vec![0.0f64; n];
    let mut a_pad = vec![0.0f64; n];
    for (&v, v_dst) in b.iter().zip(b_pad.iter_mut()) {
        *v_dst = v;
    }
    for (&v, v_dst) in a.iter().zip(a_pad.iter_mut()) {
        *v_dst = v;
    }

    let a0 = a_pad[0];
    let m = n - 1;

    // zi satisfies: zi = A*zi + b[1:] - a[1:]*b[0]/a[0]
    let mut zi = vec![0.0f64; m];

    // Direct implementation of scipy's lfilter_zi
    let rcp_a0 = 1. / a0;
    let b_pad_zero = b_pad[0];
    let _b0 = b_pad_zero / a0;
    let b_pad_iter = b_pad[1..].iter();
    let a_pad_iter = a_pad[1..].iter();
    let scale = a0 * b_pad_zero * a0;
    for ((dst, &b_pad), &a_pad) in zi[..m].iter_mut().zip(b_pad_iter).zip(a_pad_iter) {
        *dst = fmla(b_pad, rcp_a0, -a_pad * scale);
    }
    // Accumulate
    for i in (1..m).rev() {
        zi[i - 1] += zi[i];
    }

    zi
}

fn lfilter_with_zi(b: &[f64], a: &[f64], x: &[f64], zi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let m = b.len().max(a.len()) - 1;
    let mut y = vec![0.0f64; n];
    let mut z = zi.to_vec();
    z.resize(m.max(1), 0.0); // ensure at least 1 element to avoid index panic

    let a0 = a[0];

    let mut b_full = vec![0.0f64; m + 1];
    let mut a_full = vec![0.0f64; m + 1];

    let rcp_a0 = 1. / a0;

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

    (y, z)
}

pub(crate) fn filtfilt_impl(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    filter_padding: FilterPadding,
) -> Result<Vec<f64>, FiltfiltError> {
    if b.is_empty() {
        return Err(FiltfiltError::EmptyNumerator);
    }
    if a.is_empty() {
        return Err(FiltfiltError::EmptyDenominator);
    }
    if a[0] == 0.0 || !a[0].is_finite() {
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
    if (a[0] - 1.0).abs() > f64::EPSILON {
        let a0 = a[0];
        let rcp_a0 = 1. / a0;
        b = std::borrow::Cow::Owned(b.iter().map(|&v| v * rcp_a0).collect::<Vec<_>>());
        a = std::borrow::Cow::Owned(a.iter().map(|&v| v * rcp_a0).collect::<Vec<_>>());
    };

    let n = x.len();
    let pad = 3 * (b.len().max(a.len()));

    // Edge padding (reflect)
    let padded = filter_padding.extend(x, pad);

    // Initial conditions scaled to first sample
    let zi_base = lfilter_zi(b.as_ref(), a.as_ref());
    let zi: Vec<f64> = zi_base.iter().map(|&z| z * padded[0]).collect();

    // Forward pass
    let (forward, _) = lfilter_with_zi(b.as_ref(), a.as_ref(), &padded, &zi);

    // Reverse + backward pass
    let rev: Vec<f64> = forward.into_iter().rev().collect();
    let zi_back: Vec<f64> = zi_base.iter().map(|&z| z * rev[0]).collect();
    let (backward, _) = lfilter_with_zi(b.as_ref(), a.as_ref(), &rev, &zi_back);

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
        let zi = lfilter_zi(&[1.0], &[1.0]);
        assert!(zi.is_empty());
    }

    #[test]
    fn test_lfilter_zi_simple_lowpass() {
        // Simple first-order: b=[0.5, 0.5], a=[1.0, -0.0]
        // scipy: lfilter_zi([0.5,0.5],[1.0,0.0]) == [0.5]
        let zi = lfilter_zi(&[0.5, 0.5], &[1.0, 0.0]);
        assert_eq!(zi.len(), 1);
        assert!((zi[0] - 0.5).abs() < 1e-10, "zi[0] = {}", zi[0]);
    }

    #[test]
    fn test_lfilter_zi_length() {
        // zi length should be max(len(b), len(a)) - 1
        let b = vec![1.0, 2.0, 3.0];
        let a = vec![1.0, 0.5, 0.25];
        let zi = lfilter_zi(&b, &a);
        assert_eq!(zi.len(), 2);
    }

    #[test]
    fn test_lfilter_zi_no_nans() {
        let b = vec![0.1, 0.2, 0.1];
        let a = vec![1.0, -0.5, 0.1];
        let zi = lfilter_zi(&b, &a);
        no_nans(&zi, "lfilter_zi");
    }

    // ------------------------------------------------------------------
    // lfilter_with_zi
    // ------------------------------------------------------------------

    #[test]
    fn test_lfilter_with_zi_passthrough() {
        // b=[1], a=[1], zi=[] → output == input
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (y, _) = lfilter_with_zi(&[1.0], &[1.0], &x, &[]);
        assert_vec_approx(&y, &x, TOL, "passthrough");
    }

    #[test]
    fn test_lfilter_with_zi_constant_signal_settled() {
        // With correct zi a constant signal should pass through unchanged
        let b = vec![0.5, 0.5];
        let a = vec![1.0, 0.0];
        let x = vec![3.0; 20];
        let zi_base = lfilter_zi(&b, &a);
        let zi: Vec<f64> = zi_base.iter().map(|&z| z * x[0]).collect();
        let (y, _) = lfilter_with_zi(&b, &a, &x, &zi);
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
        let (y, zf) = lfilter_with_zi(&b, &a, &x, &zi);
        assert_eq!(y.len(), 100);
        assert_eq!(zf.len(), 2);
    }

    #[test]
    fn test_lfilter_with_zi_no_nans() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let b = vec![0.2, 0.5, 0.2];
        let a = vec![1.0, -0.3, 0.1];
        let zi = vec![0.0; 2];
        let (y, _) = lfilter_with_zi(&b, &a, &x, &zi);
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
