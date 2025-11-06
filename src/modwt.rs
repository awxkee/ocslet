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
use crate::convolve1d::Convolve1d;
use crate::err::{OscletError, try_vec};
use crate::util::Wavelet;
use crate::{Dwt, MultiDwt};
use num_traits::AsPrimitive;
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg};

/// Trait for performing Maximal Overlap Discrete Wavelet Transform (MODWT) on 1D signals.
///
/// # Type Parameters
/// - `T`: The numeric type of the input signal (e.g., `f32` or `f64`).
pub trait MoDwtExecutor<T> {
    /// Computes a single-level MODWT decomposition of the input signal.
    ///
    /// # Parameters
    /// - `input`: Slice containing the input signal.
    /// - `level`: The decomposition level (usually starting from 1).
    ///
    /// # Returns
    /// A `Result` containing a `Dwt<T>` with the wavelet and scaling coefficients
    /// at the specified level, or an `OscletError` if computation fails.
    fn modwt(&self, input: &[T], level: usize) -> Result<Dwt<T>, OscletError>;

    /// Computes a multi-level MODWT decomposition of the input signal.
    ///
    /// # Parameters
    /// - `input`: Slice containing the input signal.
    /// - `level`: The number of decomposition levels to compute.
    ///
    /// # Returns
    /// A `Result` containing a `MultiDwt<T>` representing all levels of wavelet
    /// and scaling coefficients, or an `OscletError` if computation fails.
    fn multi_modwt(&self, input: &[T], level: usize) -> Result<MultiDwt<T>, OscletError>;

    /// Performs a single-level inverse Maximal Overlap Discrete Wavelet Transform (IMODWT).
    ///
    /// This method reconstructs the signal from one decomposition level of approximation
    /// and detail coefficients stored in a [`Dwt`] structure.
    ///
    /// # Parameters
    /// - `input`: The [`Dwt`] structure containing approximation and detail coefficients for the given level.
    /// - `output`: Mutable slice that will hold the reconstructed signal. Must have the same length as the input signal.
    /// - `scratch`: Temporary working buffer used for intermediate computations to avoid additional allocations, must match output size.
    /// - `level`: The decomposition level to reconstruct (starting from 1).
    ///
    /// # Returns
    /// - `Ok(())` if the reconstruction succeeds.
    /// - `Err(OscletError)` if buffer sizes or configuration are invalid.
    ///
    /// # Notes
    /// This function performs a single reconstruction step. To fully reconstruct
    /// a multi-level MODWT, use [`Self::imodwt`] or [`Self::multi_imodwt`].
    fn imodwt_execute(
        &self,
        input: &Dwt<T>,
        output: &mut [T],
        scratch: &mut [T],
        level: usize,
    ) -> Result<(), OscletError>;

    /// Performs a full multi-level inverse MODWT reconstruction.
    ///
    /// This method reconstructs the original signal from all decomposition levels
    /// of a single MODWT decomposition stored in a [`Dwt`] structure.
    ///
    /// # Parameters
    /// - `input`: The [`Dwt`] structure containing all decomposition levels.
    /// - `levels`: The number of levels to reconstruct. Typically matches the number of levels used in decomposition.
    ///
    /// # Returns
    /// - `Ok(Vec<T>)` containing the fully reconstructed signal.
    /// - `Err(OscletError)` if the reconstruction fails.
    ///
    /// # Notes
    /// This is a convenience method that internally calls [`Self::imodwt_execute`]
    /// in reverse order across all levels.
    fn imodwt(&self, input: &Dwt<T>, levels: usize) -> Result<Vec<T>, OscletError>;

    /// Performs a full inverse reconstruction from a multi-resolution MODWT decomposition.
    ///
    /// This method takes a [`MultiDwt`] structure, containing approximations and details
    /// at each level, and reconstructs the full-length time-domain signal.
    ///
    /// # Parameters
    /// - `input`: The [`MultiDwt`] structure containing multiple decomposition levels.
    ///
    /// # Returns
    /// - `Ok(Vec<T>)` containing the reconstructed signal.
    /// - `Err(OscletError)` if any of the internal levels are inconsistent or invalid.
    ///
    /// # Notes
    /// Unlike [`Self::imodwt`], this version is designed for multi-level, explicitly
    /// separated decompositions where each level is stored independently.
    fn multi_imodwt(&self, input: &MultiDwt<T>) -> Result<Vec<T>, OscletError>;

    /// Returns the length of the wavelet filter used by this executor.
    ///
    /// # Returns
    /// The number of coefficients in the underlying wavelet filter.
    fn filter_length(&self) -> usize;
}

pub(crate) struct MoDwtHandler<T> {
    pub(crate) h: Vec<T>,
    pub(crate) g: Vec<T>,
    pub(crate) convolution: Box<dyn Convolve1d<T> + Send + Sync>,
}

impl<T: Mul<T, Output = T> + Copy + Sqrt2Provider<T>> MoDwtHandler<T> {
    pub(crate) fn new(
        wavelet: Wavelet<T>,
        convolution: Box<dyn Convolve1d<T> + Send + Sync>,
    ) -> Self {
        let h = wavelet
            .dec_hi
            .iter()
            .map(|&x| x * T::FRAC_SQRT2)
            .collect::<Vec<_>>();

        let g = wavelet
            .dec_lo
            .iter()
            .map(|&x| x * T::FRAC_SQRT2)
            .collect::<Vec<_>>();

        Self { h, g, convolution }
    }
}

impl<T: Copy + 'static + Debug + Default + Mul<T, Output = T>> MoDwtHandler<T>
where
    f64: AsPrimitive<T>,
{
    fn circular_convolve(
        &self,
        input: &[T],
        output: &mut [T],
        kernel: &[T],
        level: usize,
    ) -> Result<(), OscletError> {
        let stride = 1usize << (level.saturating_sub(1));
        let mut new_kernel = try_vec![T::default(); kernel.len() * stride];

        for (&h, dst) in kernel.iter().zip(new_kernel.iter_mut().step_by(stride)) {
            *dst = h;
        }

        new_kernel = new_kernel.iter().copied().rev().collect();

        self.convolution
            .convolve(input, output, &new_kernel, -(new_kernel.len() as isize / 2))?;

        Ok(())
    }
}

impl<T: Copy + 'static + Debug + Default + Mul<T, Output = T> + Add<T, Output = T>> MoDwtHandler<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(clippy::too_many_arguments)]
    fn circular_convolve_synthesize(
        &self,
        input_h: &[T],
        input_g: &[T],
        output: &mut [T],
        scratch: &mut [T],
        h_kernel: &[T],
        g_kernel: &[T],
        level: usize,
    ) -> Result<(), OscletError> {
        let stride = 1usize << (level.saturating_sub(1));
        let mut new_h_kernel = try_vec![T::default(); h_kernel.len() * stride];
        let mut new_g_kernel = try_vec![T::default(); g_kernel.len() * stride];

        for (((&h, &g), h_dst), g_dst) in h_kernel
            .iter()
            .zip(g_kernel.iter())
            .zip(new_h_kernel.iter_mut().step_by(stride))
            .zip(new_g_kernel.iter_mut().step_by(stride))
        {
            *h_dst = h;
            *g_dst = g;
        }

        // filters here is supposed to be reversed ones

        let filter_center_offset_h = (new_h_kernel.len() as isize - 1) / 2;
        let filter_center_offset_g = (new_g_kernel.len() as isize - 1) / 2;

        self.convolution
            .convolve(input_h, output, &new_h_kernel, filter_center_offset_h)?;
        self.convolution
            .convolve(input_g, scratch, &new_g_kernel, filter_center_offset_g)?;

        for (dst, src) in output.iter_mut().zip(scratch.iter()) {
            *dst = *dst + *src;
        }

        Ok(())
    }
}

impl<
    T: Copy + 'static + Debug + Default + Mul<T, Output = T> + Neg<Output = T> + Add<T, Output = T>,
> MoDwtExecutor<T> for MoDwtHandler<T>
where
    f64: AsPrimitive<T>,
{
    fn modwt(&self, input: &[T], level: usize) -> Result<Dwt<T>, OscletError> {
        let mut approx = try_vec![T::default(); input.len()];
        let mut details = try_vec![T::default(); input.len()];

        if level == 0 {
            self.circular_convolve(input, &mut approx, &self.h, level + 1)?;
            self.circular_convolve(input, &mut details, &self.g, level + 1)?;

            Ok(Dwt {
                approximations: approx,
                details,
            })
        } else {
            let mut input_proxy = input.to_vec();

            for j in 0..level {
                self.circular_convolve(&input_proxy, &mut approx, &self.h, j + 1)?;
                self.circular_convolve(&input_proxy, &mut details, &self.g, j + 1)?;

                input_proxy = details.to_vec();
            }

            Ok(Dwt {
                approximations: approx,
                details,
            })
        }
    }

    fn multi_modwt(&self, input: &[T], level: usize) -> Result<MultiDwt<T>, OscletError> {
        let mut approx = try_vec![T::default(); input.len()];
        let mut details = try_vec![T::default(); input.len()];

        if level == 0 {
            self.circular_convolve(input, &mut approx, &self.h, level + 1)?;
            self.circular_convolve(input, &mut details, &self.g, level + 1)?;

            Ok(MultiDwt {
                levels: vec![Dwt {
                    approximations: approx,
                    details,
                }],
            })
        } else {
            let mut delegated_input = input.to_vec();

            let mut levels = Vec::with_capacity(level);

            for j in 0..level {
                self.circular_convolve(&delegated_input, &mut approx, &self.h, j + 1)?;
                self.circular_convolve(&delegated_input, &mut details, &self.g, j + 1)?;

                levels.push(Dwt {
                    approximations: approx.to_vec(),
                    details: details.to_vec(),
                });

                delegated_input = details.to_vec();
            }

            Ok(MultiDwt { levels })
        }
    }

    fn imodwt_execute(
        &self,
        input: &Dwt<T>,
        output: &mut [T],
        scratch: &mut [T],
        level: usize,
    ) -> Result<(), OscletError> {
        if input.details.len() != input.approximations.len() {
            return Err(OscletError::ApproxDetailsNotMatches(
                input.approximations.len(),
                input.details.len(),
            ));
        }

        self.circular_convolve_synthesize(
            &input.approximations,
            &input.details,
            output,
            scratch,
            &self.h,
            &self.g,
            level,
        )?;

        Ok(())
    }

    fn imodwt(&self, input: &Dwt<T>, levels: usize) -> Result<Vec<T>, OscletError> {
        if input.details.len() != input.approximations.len() {
            return Err(OscletError::ApproxDetailsNotMatches(
                input.approximations.len(),
                input.details.len(),
            ));
        }
        let mut output = try_vec![T::default(); input.details.len()];
        let mut scratch = try_vec![T::default(); output.len()];
        for j in (0..levels.max(1)).rev() {
            self.imodwt_execute(input, &mut output, &mut scratch, j + 1)?;
        }
        Ok(output)
    }

    fn multi_imodwt(&self, input: &MultiDwt<T>) -> Result<Vec<T>, OscletError> {
        if input.levels.is_empty() {
            return Ok(vec![]);
        }
        let base_len = input.levels[0].details.len();
        let mut output = try_vec![T::default(); input.levels[0].details.len()];
        let mut scratch = try_vec![T::default(); output.len()];
        for j in (0..input.levels.len()).rev() {
            let level = &input.levels[j];
            let h = &level.approximations;
            let g = &level.details;
            if h.len() != g.len() || h.len() != base_len {
                return Err(OscletError::ApproxDetailsNotMatches(h.len(), g.len()));
            }
            self.imodwt_execute(level, &mut output, &mut scratch, j + 1)?;
        }
        Ok(output)
    }

    fn filter_length(&self) -> usize {
        self.h.len()
    }
}

pub(crate) trait Sqrt2Provider<T> {
    const FRAC_SQRT2: T;
}

impl Sqrt2Provider<f32> for f32 {
    const FRAC_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
}

impl Sqrt2Provider<f64> for f64 {
    const FRAC_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolve1d::ConvolveFactory;
    use crate::util::fill_wavelet;
    use crate::{BorderMode, DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_modwt() {
        const R: [f64; 16] = [
            0.2933012701892219,
            -0.34330127018922196,
            -0.11895825622994272,
            0.0,
            1.0245190528383288,
            -0.06698729810778081,
            -0.0915063509461096,
            -0.6830127018922193,
            -0.3196152422706633,
            -0.9586696879329399,
            2.5532849302036027,
            -2.0158493649053892,
            1.0346315822562817,
            1.6369546181365429,
            -1.2256569860407205,
            -0.7191342951089922,
        ];

        const G: [f64; 16] = [
            0.6151923788646684,
            0.9151923788646683,
            1.0779328524455039,
            1.6339745962155612,
            2.90849364905389,
            3.6160254037844384,
            2.707531754730548,
            1.1830127018922192,
            0.2803847577293368,
            0.3770998275257337,
            2.571715069796397,
            4.45915063509461,
            4.177932852445504,
            5.374519052838329,
            5.310816684934151,
            2.091025403784439,
        ];

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];

        let handler = MoDwtHandler::new(
            fill_wavelet(&DaubechiesFamily::Db2.get_wavelet()).unwrap(),
            f64::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.modwt(&input, 0).unwrap();
        println!("approx {:?}", result.approximations);
        println!("details {:?}", result.details);
        result.approximations.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (R[i] - x).abs() < 1e-7,
                "approximations difference expected to be < 1e-7, but values were ref {}, derived {}",
                R [i],
                x
            );
        });
        result.details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (G[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                G[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_db3() {
        const R: [f64; 16] = [
            0.7602005573613583,
            -0.20460444229355834,
            0.031279535771333355,
            0.11590406020442046,
            0.7380821865052201,
            -0.5355073542867561,
            -0.23060956154468737,
            -0.43495985349740746,
            0.13099780857472543,
            -0.37462702267183434,
            2.2930728685639945,
            -2.4185462121486423,
            1.1972919390341685,
            0.9679601951536156,
            -1.7703153342661122,
            -0.265619370459838,
        ];

        const G: [f64; 16] = [
            4.566142756803549,
            1.5934350905820895,
            0.7075834751459268,
            0.9119775689394413,
            1.0484786032652609,
            1.874105730934265,
            3.2315220628602686,
            3.5677411859496435,
            2.406598727191949,
            0.9624666820881184,
            -0.0016288718513219717,
            0.7448769571356894,
            3.2320012886407072,
            4.143420004920055,
            4.4593019579436595,
            5.851976779450698,
        ];

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];

        let handler = MoDwtHandler::new(
            fill_wavelet(&DaubechiesFamily::Db3.get_wavelet()).unwrap(),
            f64::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.modwt(&input, 0).unwrap();
        result.approximations.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (R[i] - x).abs() < 1e-7,
                "approximations difference expected to be < 1e-7, but values were ref {}, derived {}",
                R [i],
                x
            );
        });
        result.details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (G[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                G[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_db4() {
        const R: [f64; 16] = [
            0.9820692184196739,
            -0.38419579610469146,
            -0.0392393098188637,
            0.08274054649875051,
            0.46269495045245645,
            -0.7433775852034066,
            -0.058493625244623426,
            -0.10813511967149075,
            0.3617186081470942,
            -0.2314441541587134,
            1.7495123518783358,
            -2.6825617986200347,
            1.5204016306132757,
            0.516923410933862,
            -1.8286638244473488,
            0.40005049632572476,
        ];

        const G: [f64; 16] = [
            4.9784271443926595,
            5.904326868197376,
            3.802697722440679,
            1.2988430347048523,
            0.7895575300847464,
            0.8745725710108443,
            1.0879364509441398,
            2.1828050722191965,
            3.460061080601454,
            3.4119139535881344,
            2.1618909334865553,
            0.6436791036056402,
            -0.13595163054590922,
            1.3860710148361446,
            3.5053807685956766,
            3.947788381837811,
        ];

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];

        let handler = MoDwtHandler::new(
            fill_wavelet(&DaubechiesFamily::Db4.get_wavelet()).unwrap(),
            f64::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.modwt(&input, 0).unwrap();
        result.approximations.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (R[i] - x).abs() < 1e-7,
                "approximations difference expected to be < 1e-7, but values were ref {}, derived {}",
                R [i],
                x
            );
        });
        result.details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (G[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                G[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_db3_level() {
        const R: [f64; 16] = [
            -0.7572385310039746,
            0.14621490277526755,
            0.6015285930082527,
            0.5012447447536913,
            -0.39337009236090054,
            -1.1949192562149717,
            -1.310574464440232,
            -0.9687514441521241,
            0.35229038297998505,
            1.6332941447543012,
            2.0139256992513395,
            1.9203164675168596,
            0.7983182403848902,
            -0.5845897913145967,
            -1.3048798278193603,
            -1.4528097681184267,
        ];

        const G: [f64; 16] = [
            1.860405833627104,
            2.1510852675498118,
            2.4534026816895635,
            2.698279161983151,
            3.0110144569919797,
            3.1997896209895167,
            3.162448553121543,
            3.1657858613815844,
            3.0520941663728953,
            2.7614147324501874,
            2.459097318310435,
            2.2142208380168484,
            1.9014855430080195,
            1.712710379010483,
            1.750051446878456,
            1.746714138618416,
        ];

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];

        let handler = MoDwtHandler::new(
            fill_wavelet(&DaubechiesFamily::Db3.get_wavelet()).unwrap(),
            f64::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.modwt(&input, 3).unwrap();
        result.approximations.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (R[i] - x).abs() < 1e-7,
                "approximations difference expected to be < 1e-7, but values were ref {}, derived {}",
                R [i],
                x
            );
        });
        result.details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (G[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                G[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_db3_level_odd() {
        const R: [f64; 17] = [
            0.8864668484721063,
            -0.7918190686469812,
            -0.037439105232348835,
            0.20833038508541157,
            0.7679726863473504,
            -0.5355073542867561,
            -0.23060956154468737,
            -0.43495985349740746,
            0.13099780857472543,
            -0.37462702267183434,
            2.2930728685639945,
            -2.4185462121486423,
            1.1972919390341685,
            0.9679601951536156,
            -1.7703153342661122,
            -0.265619370459838,
            0.4073501515232355,
        ];

        const G: [f64; 17] = [
            1.4779021844808506,
            0.5998900177564377,
            1.5307258814050357,
            1.7848346015519732,
            1.3307589279357592,
            1.874105730934265,
            3.2315220628602686,
            3.5677411859496435,
            2.406598727191949,
            0.9624666820881184,
            -0.0016288718513219717,
            0.7448769571356894,
            3.2320012886407072,
            4.143420004920055,
            4.4593019579436595,
            5.851976779450698,
            4.603505881606212,
        ];

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];

        let handler = MoDwtHandler::new(
            fill_wavelet(&DaubechiesFamily::Db3.get_wavelet()).unwrap(),
            f64::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.modwt(&input, 0).unwrap();
        result.approximations.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (R[i] - x).abs() < 1e-7,
                "approximations difference expected to be < 1e-7, but values were ref {}, derived {}",
               R [i],
                x
            );
        });
        result.details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (G[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                G[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_db2_level_odd() {
        const R: [f64; 17] = [
            0.7500000000000001,
            -0.5000000000000001,
            -0.5000000000000001,
            -0.5,
            1.0000000000000002,
            0.5000000000000001,
            0.5000000000000001,
            -0.5000000000000001,
            -0.7000000000000001,
            -2.0500000000000007,
            2.0500000000000007,
            -2.000000000000001,
            0.6000000000000004,
            2.3000000000000003,
            0.04999999999999999,
            -0.40000000000000013,
            -0.6000000000000001,
        ];

        const G: [f64; 17] = [
            1.7500000000000004,
            1.5000000000000004,
            2.5000000000000004,
            3.500000000000001,
            3.000000000000001,
            1.5000000000000004,
            0.5000000000000001,
            0.5000000000000001,
            1.7000000000000004,
            4.450000000000001,
            4.450000000000001,
            4.400000000000001,
            5.800000000000002,
            2.900000000000001,
            0.55,
            0.9000000000000002,
            1.9000000000000004,
        ];

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];

        let handler = MoDwtHandler::new(
            fill_wavelet(&DaubechiesFamily::Db1.get_wavelet()).unwrap(),
            f64::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.modwt(&input, 0).unwrap();
        result.approximations.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (R[i] - x).abs() < 1e-7,
                "approximations difference expected to be < 1e-7, but values were ref {}, derived {}",
                R [i],
                x
            );
        });
        result.details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (G[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                G[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_round_trip() {
        const R: [f64; 16] = [
            0.2933012701892219,
            -0.34330127018922196,
            -0.11895825622994272,
            0.0,
            1.0245190528383288,
            -0.06698729810778081,
            -0.0915063509461096,
            -0.6830127018922193,
            -0.3196152422706633,
            -0.9586696879329399,
            2.5532849302036027,
            -2.0158493649053892,
            1.0346315822562817,
            1.6369546181365429,
            -1.2256569860407205,
            -0.7191342951089922,
        ];

        const G: [f64; 16] = [
            0.6151923788646684,
            0.9151923788646683,
            1.0779328524455039,
            1.6339745962155612,
            2.90849364905389,
            3.6160254037844384,
            2.707531754730548,
            1.1830127018922192,
            0.2803847577293368,
            0.3770998275257337,
            2.571715069796397,
            4.45915063509461,
            4.177932852445504,
            5.374519052838329,
            5.310816684934151,
            2.091025403784439,
        ];

        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];

        let handler = MoDwtHandler::new(
            fill_wavelet(&DaubechiesFamily::Db2.get_wavelet()).unwrap(),
            f64::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.modwt(&input, 1).unwrap();
        result.approximations.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (R[i] - x).abs() < 1e-7,
                "approximations difference expected to be < 1e-7, but values were ref {}, derived {}",
                R [i],
                x
            );
        });
        result.details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (G[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                G[i],
                x
            );
        });

        let inverse = handler.imodwt(&result, 1).unwrap();
        inverse.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-7,
                "reconstruct difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_round_trip_multi() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];

        let handler = MoDwtHandler::new(
            fill_wavelet(&DaubechiesFamily::Db2.get_wavelet()).unwrap(),
            f64::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.multi_modwt(&input, 3).unwrap();
        let inverse = handler.multi_imodwt(&result).unwrap();

        inverse.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-7,
                "reconstruct difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_round_trip_multi_f32_db2() {
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];

        let handler = MoDwtHandler::<f32>::new(
            fill_wavelet(&DaubechiesFamily::Db2.get_wavelet()).unwrap(),
            f32::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.multi_modwt(&input, 3).unwrap();
        let inverse = handler.multi_imodwt(&result).unwrap();

        inverse.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstruct difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_round_trip_multi_f32_db3() {
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5,
        ];

        let handler = MoDwtHandler::<f32>::new(
            fill_wavelet(&DaubechiesFamily::Db3.get_wavelet()).unwrap(),
            f32::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.multi_modwt(&input, 3).unwrap();
        let inverse = handler.multi_imodwt(&result).unwrap();

        inverse.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstruct difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_round_trip_multi_f32_db4() {
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5,
        ];

        let handler = MoDwtHandler::<f32>::new(
            fill_wavelet(&DaubechiesFamily::Db4.get_wavelet()).unwrap(),
            f32::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.multi_modwt(&input, 3).unwrap();
        let inverse = handler.multi_imodwt(&result).unwrap();

        inverse.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstruct difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_modwt_round_trip_multi_f32_db8() {
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
            1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5,
        ];

        let handler = MoDwtHandler::<f32>::new(
            fill_wavelet(&DaubechiesFamily::Db8.get_wavelet()).unwrap(),
            f32::make_convolution_1d(BorderMode::Wrap),
        );
        let result = handler.multi_modwt(&input, 3).unwrap();
        let inverse = handler.multi_imodwt(&result).unwrap();

        inverse.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstruct difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }
}
