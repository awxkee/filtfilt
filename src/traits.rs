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
use crate::filtfilt::{lfilter_with_zi_impl, lfilter_zi_impl};
use crate::filtfilt_error::FiltfiltError;
use crate::sos::{sosfilt_impl, sosfilt_zi_impl, sosfiltfilt_impl};
use crate::{
    FilterPadding, LFilterBuilder, LFilterState, SosFilter, SosFilterBuilder, SosFilterState,
};
use num_traits::{Float, MulAdd};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign};

pub(crate) trait FilterSample:
    Float
    + 'static
    + Copy
    + for<'a> Sum<&'a Self>
    + MulAdd<Self, Output = Self>
    + AddAssign
    + Debug
    + MulAssign
    + Default
{
}

impl FilterSample for f32 {}
impl FilterSample for f64 {}

pub trait Filtering: Sized {
    fn lfilter_zi(b: &[Self], a: &[Self]) -> Result<Vec<Self>, FiltfiltError>;
    fn lfilter_with_zi(
        x: &[Self],
        options: LFilterBuilder<'_, Self>,
    ) -> Result<LFilterState<Self>, FiltfiltError>;
    fn sosfilt_zi(sos: &[SosFilter<Self>]) -> Result<Vec<[Self; 2]>, FiltfiltError>;
    fn sosfiltfilt(
        sos: &[SosFilter<Self>],
        x: &[Self],
        padding: FilterPadding,
    ) -> Result<Vec<Self>, FiltfiltError>;
    fn sosfilt(
        x: &[Self],
        options: SosFilterBuilder<'_, Self>,
    ) -> Result<SosFilterState<Self>, FiltfiltError>;
}

impl Filtering for f32 {
    fn lfilter_zi(b: &[Self], a: &[Self]) -> Result<Vec<Self>, FiltfiltError> {
        lfilter_zi_impl(b, a)
    }

    fn lfilter_with_zi(
        x: &[Self],
        options: LFilterBuilder<'_, Self>,
    ) -> Result<LFilterState<Self>, FiltfiltError> {
        lfilter_with_zi_impl(x, options)
    }

    fn sosfilt_zi(sos: &[SosFilter<Self>]) -> Result<Vec<[Self; 2]>, FiltfiltError> {
        sosfilt_zi_impl(sos)
    }

    fn sosfiltfilt(
        sos: &[SosFilter<Self>],
        x: &[Self],
        padding: FilterPadding,
    ) -> Result<Vec<Self>, FiltfiltError> {
        sosfiltfilt_impl(sos, x, padding)
    }

    fn sosfilt(
        x: &[Self],
        options: SosFilterBuilder<'_, Self>,
    ) -> Result<SosFilterState<Self>, FiltfiltError> {
        sosfilt_impl(x, options)
    }
}

impl Filtering for f64 {
    fn lfilter_zi(b: &[Self], a: &[Self]) -> Result<Vec<Self>, FiltfiltError> {
        lfilter_zi_impl(b, a)
    }

    fn lfilter_with_zi(
        x: &[Self],
        options: LFilterBuilder<'_, Self>,
    ) -> Result<LFilterState<Self>, FiltfiltError> {
        lfilter_with_zi_impl(x, options)
    }

    fn sosfilt_zi(sos: &[SosFilter<Self>]) -> Result<Vec<[Self; 2]>, FiltfiltError> {
        sosfilt_zi_impl(sos)
    }

    fn sosfiltfilt(
        sos: &[SosFilter<Self>],
        x: &[Self],
        padding: FilterPadding,
    ) -> Result<Vec<Self>, FiltfiltError> {
        sosfiltfilt_impl(sos, x, padding)
    }

    fn sosfilt(
        x: &[Self],
        options: SosFilterBuilder<'_, Self>,
    ) -> Result<SosFilterState<Self>, FiltfiltError> {
        sosfilt_impl(x, options)
    }
}
