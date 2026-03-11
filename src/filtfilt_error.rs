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

#[derive(Debug)]
pub enum FiltfiltError {
    EmptyNumerator,
    EmptyDenominator,
    DenominatorLeadingZero,
    NonFiniteCoefficients,
    EmptySignal,
    SignalTooShort { signal_len: usize, required: usize },
}

impl std::fmt::Display for FiltfiltError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FiltfiltError::EmptyNumerator => write!(f, "b coefficients must not be empty"),
            FiltfiltError::EmptyDenominator => write!(f, "a coefficients must not be empty"),
            FiltfiltError::DenominatorLeadingZero => {
                write!(
                    f,
                    "a[0] must not be zero or non-finite (used as normalisation divisor)"
                )
            }
            FiltfiltError::NonFiniteCoefficients => {
                write!(f, "all filter coefficients must be finite (no NaN or Inf)")
            }
            FiltfiltError::EmptySignal => write!(f, "input signal must not be empty"),
            FiltfiltError::SignalTooShort {
                signal_len,
                required,
            } => write!(
                f,
                "input signal length ({}) must be greater than padding length ({}); \
                 provide a longer signal or reduce filter order",
                signal_len, required
            ),
        }
    }
}

impl std::error::Error for FiltfiltError {}
