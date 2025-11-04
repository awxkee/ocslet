/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::WaveletFilterProvider;
use num_traits::AsPrimitive;
use std::borrow::Cow;

/// Represents the Symlet wavelet family.
///
/// Symlets are a modified version of Daubechies wavelets designed to be as
/// symmetrical as possible while retaining the same number of vanishing moments.
/// They are often used in signal processing and discrete wavelet transforms (DWT)
/// where near-symmetry is desired for better reconstruction and reduced phase distortion.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum SymletFamily {
    /// Symlet wavelet with 2 vanishing moments
    Sym2,
    /// Symlet wavelet with 3 vanishing moments
    Sym3,
    /// Symlet wavelet with 4 vanishing moments
    Sym4,
    /// Symlet wavelet with 5 vanishing moments
    Sym5,
    /// Symlet wavelet with 6 vanishing moments
    Sym6,
    /// Symlet wavelet with 7 vanishing moments
    Sym7,
    /// Symlet wavelet with 8 vanishing moments
    Sym8,
    /// Symlet wavelet with 9 vanishing moments
    Sym9,
    /// Symlet wavelet with 10 vanishing moments
    Sym10,
}

impl SymletFamily {
    pub(crate) fn get_wavelet_impl(self) -> &'static [f64] {
        match self {
            SymletFamily::Sym2 => [
                0.48296291314469025,
                0.83651630373746899,
                0.22414386804185735,
                -0.12940952255092145,
            ]
            .as_slice(),
            SymletFamily::Sym3 => [
                0.33267055295095688,
                0.80689150931333875,
                0.45987750211933132,
                -0.13501102001039084,
                -0.085441273882241486,
                0.035226291882100656,
            ]
            .as_slice(),
            SymletFamily::Sym4 => [
                0.032223100604042702,
                -0.012603967262037833,
                -0.099219543576847216,
                0.29785779560527736,
                0.80373875180591614,
                0.49761866763201545,
                -0.02963552764599851,
                -0.075765714789273325,
            ]
            .as_slice(),
            SymletFamily::Sym5 => [
                0.019538882735286728,
                -0.021101834024758855,
                -0.17532808990845047,
                0.016602105764522319,
                0.63397896345821192,
                0.72340769040242059,
                0.1993975339773936,
                -0.039134249302383094,
                0.029519490925774643,
                0.027333068345077982,
            ]
            .as_slice(),
            SymletFamily::Sym6 => [
                -0.007800708325034148,
                0.0017677118642428036,
                0.044724901770665779,
                -0.021060292512300564,
                -0.072637522786462516,
                0.3379294217276218,
                0.787641141030194,
                0.49105594192674662,
                -0.048311742585632998,
                -0.11799011114819057,
                0.0034907120842174702,
                0.015404109327027373,
            ]
            .as_slice(),
            SymletFamily::Sym7 => [
                0.010268176708511255,
                0.0040102448715336634,
                -0.10780823770381774,
                -0.14004724044296152,
                0.28862963175151463,
                0.76776431700316405,
                0.5361019170917628,
                0.017441255086855827,
                -0.049552834937127255,
                0.067892693501372697,
                0.03051551316596357,
                -0.01263630340325193,
                -0.0010473848886829163,
                0.0026818145682578781,
            ]
            .as_slice(),
            SymletFamily::Sym8 => [
                0.0018899503327594609,
                -0.0003029205147213668,
                -0.014952258337048231,
                0.0038087520138906151,
                0.049137179673607506,
                -0.027219029917056003,
                -0.051945838107709037,
                0.3644418948353314,
                0.77718575170052351,
                0.48135965125837221,
                -0.061273359067658524,
                -0.14329423835080971,
                0.0076074873249176054,
                0.031695087811492981,
                -0.00054213233179114812,
                -0.0033824159510061256,
            ]
            .as_slice(),
            SymletFamily::Sym9 => [
                0.0010694900329086053,
                -0.00047315449868008311,
                -0.010264064027633142,
                0.0088592674934004842,
                0.06207778930288603,
                -0.018233770779395985,
                -0.19155083129728512,
                0.035272488035271894,
                0.61733844914093583,
                0.717897082764412,
                0.238760914607303,
                -0.054568958430834071,
                0.00058346274612580684,
                0.03022487885827568,
                -0.01152821020767923,
                -0.013271967781817119,
                0.00061978088898558676,
                0.0014009155259146807,
            ]
            .as_slice(),
            SymletFamily::Sym10 => [
                -0.00045932942100465878,
                5.7036083618494284e-005,
                0.0045931735853118284,
                -0.00080435893201654491,
                -0.02035493981231129,
                0.0057649120335819086,
                0.049994972077376687,
                -0.0319900568824278,
                -0.035536740473817552,
                0.38382676106708546,
                0.7695100370211071,
                0.47169066693843925,
                -0.070880535783243853,
                -0.15949427888491757,
                0.011609893903711381,
                0.045927239231092203,
                -0.0014653825813050513,
                -0.0086412992770224222,
                9.5632670722894754e-005,
                0.00077015980911449011,
            ]
            .as_slice(),
        }
    }
}

impl<T: Copy + 'static> WaveletFilterProvider<T> for SymletFamily
where
    f64: AsPrimitive<T>,
{
    fn get_wavelet(&self) -> Cow<'_, [T]> {
        Cow::Owned(self.get_wavelet_impl().iter().map(|x| x.as_()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symlet_test() {
        let to_test = [
            SymletFamily::Sym2,
            SymletFamily::Sym3,
            SymletFamily::Sym4,
            SymletFamily::Sym5,
            SymletFamily::Sym6,
            SymletFamily::Sym7,
            SymletFamily::Sym8,
            SymletFamily::Sym9,
            SymletFamily::Sym10,
        ];
        for b in to_test.iter() {
            let wv: Cow<[f64]> = b.get_wavelet();
            assert!(
                wv.len().is_multiple_of(2),
                "Assertion failed for symlet {:?} with size {}",
                b,
                wv.len()
            );
        }
    }
}
