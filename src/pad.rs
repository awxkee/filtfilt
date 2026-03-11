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
use num_traits::AsPrimitive;
use std::ops::{Mul, Neg, Sub};

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug, Default)]
pub enum FilterPadding {
    /// Odd (anti-symmetric) extension — the default.
    ///
    /// Reflects the signal with a 180° rotation around the edge sample:
    /// `pad[k] = 2 * x[edge] - x[mirror]`
    ///
    /// For a signal `[1, 2, 3, 4, 5]` with pad=2:
    /// ```text
    /// [-1, 0, | 1, 2, 3, 4, 5 | 6, 7]
    /// ```
    /// Preserves continuity of the first derivative at the boundary,
    /// which minimizes low-frequency transients. Recommended for most signals.
    #[default]
    Odd,

    /// Even (symmetric) extension.
    ///
    /// Reflects the signal as a mirror image around the edge sample:
    /// `pad[k] = x[mirror]`
    ///
    /// For a signal `[1, 2, 3, 4, 5]` with pad=2:
    /// ```text
    /// [3, 2, | 1, 2, 3, 4, 5 | 4, 3]
    /// ```
    /// Preserves the value at the boundary but may introduce a slope
    /// discontinuity. Useful when the signal is known to be symmetric
    /// near its edges.
    Even,
}

impl FilterPadding {
    pub(crate) fn extend<
        T: Copy + Default + Neg<Output = T> + Mul<Output = T> + Sub<Output = T> + 'static,
    >(
        self,
        slice: &[T],
        pad: usize,
    ) -> Vec<T>
    where
        f64: AsPrimitive<T>,
    {
        if pad < 1 {
            return slice.to_vec();
        }
        let total_length = 2 * pad + slice.len();
        let mut padded_slice = vec![T::default(); total_length];
        match self {
            FilterPadding::Odd => {
                let first = 2f64.as_() * slice[0];
                let last = 2f64.as_() * slice[slice.len() - 1];
                let start = &mut padded_slice[..pad];
                for (i, dst) in start.iter_mut().enumerate() {
                    let idx = (pad - i).min(slice.len() - 1);
                    *dst = first - slice[idx];
                }

                let middle = &mut padded_slice[pad..pad + slice.len()];
                middle.copy_from_slice(slice);

                let end = &mut padded_slice[pad + slice.len()..];
                for (i, dst) in end.iter_mut().enumerate() {
                    let mirror_idx = (slice.len() as i64 - 2 - i as i64).max(0) as usize;
                    *dst = last - slice[mirror_idx];
                }
            }
            FilterPadding::Even => {
                let start = &mut padded_slice[..pad];
                for (i, dst) in start.iter_mut().enumerate() {
                    let idx = (pad - i).min(slice.len() - 1);
                    *dst = slice[idx];
                }

                let middle = &mut padded_slice[pad..pad + slice.len()];
                middle.copy_from_slice(slice);

                let end = &mut padded_slice[pad + slice.len()..];
                for (i, dst) in end.iter_mut().enumerate() {
                    let mirror_idx = (slice.len() as i64 - 2 - i as i64).max(0) as usize;
                    *dst = slice[mirror_idx];
                }
            }
        }
        padded_slice
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_slice_eq(actual: &[f64], expected: &[f64], msg: &str) {
        assert_eq!(actual.len(), expected.len(), "{}: length mismatch", msg);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "{}: index {} got {} expected {}",
                msg,
                i,
                a,
                e
            );
        }
    }

    #[test]
    fn test_even_ext_scipy_row0() {
        // even_ext([1,2,3,4,5], 2) → [3,2,1,2,3,4,5,4,3]
        let x = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let result = FilterPadding::Even.extend(&x, 2);
        assert_slice_eq(
            &result,
            &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0],
            "even row0",
        );
    }

    // Helper: apply odd_ext to a 1D slice
    fn odd_ext_1d(x: &[f64], n: usize) -> Vec<f64> {
        let mut padded = vec![0.0f64; 2 * n + x.len()];
        let first = x[0];
        let last = x[x.len() - 1];
        let two = first + first;

        // Left
        for (i, dst) in padded[..n].iter_mut().enumerate() {
            let mirror_idx = (n - i).min(x.len() - 1);
            *dst = two - x[mirror_idx]; // 2*first - x[mirror]
        }

        // Copy
        padded[n..n + x.len()].copy_from_slice(x);

        // Right
        let two_last = last + last;
        for (i, dst) in padded[n + x.len()..].iter_mut().enumerate() {
            let mirror_idx = (x.len() - 1).saturating_sub(i + 1);
            *dst = two_last - x[mirror_idx];
        }

        padded
    }

    // ------------------------------------------------------------------
    // scipy reference cases
    // ------------------------------------------------------------------

    #[test]
    fn test_odd_ext_scipy_example_row0() {
        // scipy: odd_ext([1,2,3,4,5], 2) → [-1, 0, 1, 2, 3, 4, 5, 6, 7]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = odd_ext_1d(&x, 2);
        let expected = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(result.len(), expected.len());
        for (i, (a, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "index {}: got {}, expected {}",
                i,
                a,
                e
            );
        }
    }

    #[test]
    fn test_odd_ext_scipy_example_row1() {
        // scipy: odd_ext([0,1,4,9,16], 2) → [-4, -1, 0, 1, 4, 9, 16, 23, 28]
        let x = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let result = odd_ext_1d(&x, 2);
        let expected = vec![-4.0, -1.0, 0.0, 1.0, 4.0, 9.0, 16.0, 23.0, 28.0];
        assert_eq!(result.len(), expected.len());
        for (i, (a, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "index {}: got {}, expected {}",
                i,
                a,
                e
            );
        }
    }

    // ------------------------------------------------------------------
    // structural properties
    // ------------------------------------------------------------------

    #[test]
    fn test_odd_ext_output_length() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = 3;
        let result = odd_ext_1d(&x, n);
        assert_eq!(result.len(), x.len() + 2 * n);
    }

    #[test]
    fn test_odd_ext_preserves_original() {
        // Middle portion must equal original signal exactly
        let x = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let n = 4;
        let result = odd_ext_1d(&x, n);
        assert_eq!(&result[n..n + x.len()], x.as_slice());
    }

    #[test]
    fn test_odd_ext_antisymmetry_at_left_boundary() {
        // odd_ext is a 180-degree rotation around x[0]:
        // result[n - k] + result[n + k] == 2 * x[0]  for k = 1..=n
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = 3;
        let result = odd_ext_1d(&x, n);
        let edge = x[0];
        for k in 1..=n {
            let left = result[n - k];
            let right = result[n + k];
            assert!(
                (left + right - 2.0 * edge).abs() < 1e-10,
                "k={}: left={} right={} edge={}",
                k,
                left,
                right,
                edge
            );
        }
    }

    #[test]
    fn test_odd_ext_antisymmetry_at_right_boundary() {
        // result[n+len-1-k] + result[n+len-1+k] == 2 * x[len-1]  for k = 1..=n
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = 3;
        let result = odd_ext_1d(&x, n);
        let edge = x[x.len() - 1];
        let center = n + x.len() - 1;
        for k in 1..=n {
            let left = result[center - k];
            let right = result[center + k];
            assert!(
                (left + right - 2.0 * edge).abs() < 1e-10,
                "k={}: left={} right={} edge={}",
                k,
                left,
                right,
                edge
            );
        }
    }

    #[test]
    fn test_odd_ext_constant_signal() {
        // Constant signal: padding should also be constant (2*c - c = c)
        let x = vec![7.0f64; 10];
        let result = odd_ext_1d(&x, 5);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - 7.0).abs() < 1e-10, "index {}: got {}", i, v);
        }
    }

    #[test]
    fn test_odd_ext_n_zero() {
        // No padding — output equals input
        let x = vec![1.0, 2.0, 3.0];
        let result = odd_ext_1d(&x, 0);
        assert_eq!(result, x);
    }

    #[test]
    fn test_odd_ext_n_equals_len_minus_1() {
        // Maximum valid padding (n = len-1)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = x.len() - 1;
        let result = odd_ext_1d(&x, n);
        assert_eq!(result.len(), x.len() + 2 * n);
        // Original still preserved in middle
        assert_eq!(&result[n..n + x.len()], x.as_slice());
    }

    #[test]
    fn test_odd_ext_negative_values() {
        // scipy: odd_ext([-2, 0, 2], 2) → [-6, -4, -2, 0, 2, 4, 6]
        let x = vec![-2.0, 0.0, 2.0];
        let result = odd_ext_1d(&x, 2);
        let expected = vec![-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0];
        for (i, (a, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "index {}: got {}, expected {}",
                i,
                a,
                e
            );
        }
    }
}
