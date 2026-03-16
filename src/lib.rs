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
mod filtfilt;
mod filtfilt_error;
mod mla;
mod pad;
mod sos;
mod traits;

use crate::filtfilt::{filtfilt_impl, lfilter_with_zi_impl, lfilter_zi_impl};
use crate::filtfilt_error::FiltfiltError;
pub use crate::pad::FilterPadding;
use crate::sos::{sosfilt_impl, sosfilt_zi_impl};
pub use filtfilt::{LFilterBuilder, LFilterState};
pub use sos::{SosFilter, SosFilterBuilder, SosFilterState};
pub use traits::Filtering;

/// Configuration for a zero-phase digital filter applied via [`filtfilt`] or [`filtfilt_f32`].
///
/// Holds the IIR filter coefficients and edge-padding strategy.
/// Coefficients can be owned (`Vec`) or borrowed (`&[f64]`) via [`Cow`].
///
/// # Fields
/// - `a` — denominator (feedback) coefficients, `a[0]` is the normalization factor,
///   typically `1.0`. Length must match `b`.
/// - `b` — numerator (feedforward) coefficients.
/// - `pad_type` — edge extension strategy used before filtering to reduce
///   boundary transients. See [`FilterPadding`].
pub struct FilterOptions<'a, T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// Denominator
    pub a: std::borrow::Cow<'a, [T]>,
    /// Numerator
    pub b: std::borrow::Cow<'a, [T]>,
    pub pad_type: FilterPadding,
}

/// Applies a zero-phase forward-backward IIR filter to `x` (f64).
pub fn filtfilt(
    x: &[f64],
    filtfilt_options: FilterOptions<'_, f64>,
) -> Result<Vec<f64>, FiltfiltError> {
    filtfilt_impl(
        filtfilt_options.b.as_ref(),
        filtfilt_options.a.as_ref(),
        x,
        filtfilt_options.pad_type,
    )
}

/// Applies a zero-phase forward-backward IIR filter to `x` (f32).
///
/// Convenience wrapper around [`filtfilt`] for `f32` signals.
/// Input is widened to `f64` internally for numerical stability,
/// then the result is narrowed back to `f32` on return.
///
pub fn filtfilt_f32(
    x: &[f32],
    filtfilt_options: FilterOptions<'_, f32>,
) -> Result<Vec<f32>, FiltfiltError> {
    filtfilt_impl(
        filtfilt_options.b.as_ref(),
        filtfilt_options.a.as_ref(),
        x,
        filtfilt_options.pad_type,
    )
}

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
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Result<Vec<f64>, FiltfiltError> {
    lfilter_zi_impl(b, a)
}

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
pub fn lfilter_zi_f32(b: &[f32], a: &[f32]) -> Result<Vec<f32>, FiltfiltError> {
    lfilter_zi_impl(b, a)
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
pub fn lfilter_with_zi(
    x: &[f64],
    options: LFilterBuilder<'_, f64>,
) -> Result<LFilterState<f64>, FiltfiltError> {
    lfilter_with_zi_impl(x, options)
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
pub fn lfilter_with_zi_f32(
    x: &[f32],
    options: LFilterBuilder<'_, f32>,
) -> Result<LFilterState<f32>, FiltfiltError> {
    lfilter_with_zi_impl(x, options)
}

/// Compute the initial conditions for one biquad section (b, a already
/// normalised so a[0] == 1). Returns `[z0, z1]`.
///
/// Solves `(I - companion(a).T) * zi = b[1:] - a[1:]*b[0]` using the
/// explicit recurrence from scipy's `lfilter_zi`:
///
/// ```text
/// zi[0] = (b[1] + b[2] - (a[1] + a[2])*b[0]) / (1 + a[1] + a[2])
/// zi[1] = a[2]*zi[0] - (b[2] - a[2]*b[0])
/// ```
pub fn sosfilt_zi(sos: &[SosFilter<f64>]) -> Result<Vec<[f64; 2]>, FiltfiltError> {
    sosfilt_zi_impl(sos)
}

/// Compute the initial conditions for one biquad section (b, a already
/// normalised so a[0] == 1). Returns `[z0, z1]`.
///
/// Solves `(I - companion(a).T) * zi = b[1:] - a[1:]*b[0]` using the
/// explicit recurrence from scipy's `lfilter_zi`:
///
/// ```text
/// zi[0] = (b[1] + b[2] - (a[1] + a[2])*b[0]) / (1 + a[1] + a[2])
/// zi[1] = a[2]*zi[0] - (b[2] - a[2]*b[0])
/// ```
pub fn sosfilt_zi_f32(sos: &[SosFilter<f32>]) -> Result<Vec<[f32; 2]>, FiltfiltError> {
    sosfilt_zi_impl(sos)
}

/// Apply a cascade of second-order sections to a signal in one direction.
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
pub fn sosfilt(
    x: &[f64],
    options: SosFilterBuilder<'_, f64>,
) -> Result<SosFilterState<f64>, FiltfiltError> {
    sosfilt_impl(x, options)
}

/// Apply a cascade of second-order sections to a signal in one direction.
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
pub fn sosfilt_f32(
    x: &[f32],
    options: SosFilterBuilder<'_, f32>,
) -> Result<SosFilterState<f32>, FiltfiltError> {
    sosfilt_impl(x, options)
}

/// Apply a digital filter forward and backward over a signal using
/// second-order sections (SOS), matching scipy's `sosfiltfilt`.
///
/// The algorithm: for each `SosFilter` section run `filtfilt` with that
/// section's `(b, a)`. The output of each section feeds into the next.
pub fn sosfiltfilt(
    sos: &[SosFilter<f64>],
    x: &[f64],
    padding: FilterPadding,
) -> Result<Vec<f64>, FiltfiltError> {
    f64::sosfiltfilt(sos, x, padding)
}

/// Apply a digital filter forward and backward over a signal using
/// second-order sections (SOS), matching scipy's `sosfiltfilt`.
///
/// The algorithm: for each `SosFilter` section run `filtfilt` with that
/// section's `(b, a)`. The output of each section feeds into the next.
pub fn sosfiltfilt_f32(
    sos: &[SosFilter<f32>],
    x: &[f32],
    padding: FilterPadding,
) -> Result<Vec<f32>, FiltfiltError> {
    f32::sosfiltfilt(sos, x, padding)
}
