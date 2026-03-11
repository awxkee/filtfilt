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

use crate::filtfilt::filtfilt_impl;
use crate::filtfilt_error::FiltfiltError;
use crate::pad::FilterPadding;

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
pub struct FilterOptions<'a> {
    /// Denominator
    pub a: std::borrow::Cow<'a, [f64]>,
    /// Numerator
    pub b: std::borrow::Cow<'a, [f64]>,
    pub pad_type: FilterPadding,
}

/// Applies a zero-phase forward-backward IIR filter to `x` (f64).
pub fn filtfilt(x: &[f64], filtfilt_options: FilterOptions<'_>) -> Result<Vec<f64>, FiltfiltError> {
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
    filtfilt_options: FilterOptions<'_>,
) -> Result<Vec<f32>, FiltfiltError> {
    Ok(filtfilt_impl(
        filtfilt_options.b.as_ref(),
        filtfilt_options.a.as_ref(),
        x.iter().map(|&x| x as f64).collect::<Vec<_>>().as_ref(),
        filtfilt_options.pad_type,
    )?
    .iter()
    .map(|&x| x as f32)
    .collect::<Vec<_>>())
}
