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
use crate::BorderMode;
use crate::err::OscletError;
use crate::filter_padding::make_arena_1d;
use crate::mla::fmla;
use num_traits::{AsPrimitive, MulAdd};
use std::marker::PhantomData;
use std::ops::{Add, Mul};

pub(crate) trait Convolve1d<T> {
    fn convolve(
        &self,
        input: &[T],
        output: &mut [T],
        kernel: &[T],
        filter_center: isize,
    ) -> Result<(), OscletError>;
}

pub(crate) trait ConvolveFactory<T> {
    fn make_convolution_1d(border_mode: BorderMode) -> Box<dyn Convolve1d<T> + Send + Sync>;
}

impl ConvolveFactory<f32> for f32 {
    fn make_convolution_1d(border_mode: BorderMode) -> Box<dyn Convolve1d<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonConvolution1dF32;
            Box::new(NeonConvolution1dF32 { border_mode })
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::factory::has_valid_avx;
            if has_valid_avx() {
                use crate::avx::AvxConvolution1dF32;
                return Box::new(AvxConvolution1dF32 { border_mode });
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Box::new(ScalarConvolution1d {
                phantom_data: PhantomData,
                border_mode,
            })
        }
    }
}

impl ConvolveFactory<f64> for f64 {
    fn make_convolution_1d(border_mode: BorderMode) -> Box<dyn Convolve1d<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonConvolution1dF64;
            Box::new(NeonConvolution1dF64 { border_mode })
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::factory::has_valid_avx;
            if has_valid_avx() {
                use crate::avx::AvxConvolution1dF64;
                return Box::new(AvxConvolution1dF64 { border_mode });
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Box::new(ScalarConvolution1d {
                phantom_data: PhantomData,
                border_mode,
            })
        }
    }
}

#[allow(unused)]
pub(crate) struct ScalarConvolution1d<T> {
    phantom_data: PhantomData<T>,
    border_mode: BorderMode,
}

impl<T: Copy + 'static + MulAdd<T, Output = T> + Add<T, Output = T> + Mul<T, Output = T> + Default>
    Convolve1d<T> for ScalarConvolution1d<T>
where
    f64: AsPrimitive<T>,
{
    fn convolve(
        &self,
        input: &[T],
        output: &mut [T],
        kernel: &[T],
        filter_center: isize,
    ) -> Result<(), OscletError> {
        if input.len() != output.len() {
            return Err(OscletError::InOutSizesMismatch(input.len(), output.len()));
        }

        let filter_size = kernel.len();

        if kernel.is_empty() {
            output.copy_from_slice(input);
            return Ok(());
        }

        if filter_center.unsigned_abs() >= filter_size {
            return Err(OscletError::MisconfiguredFilterCenter(
                filter_center.unsigned_abs(),
                kernel.len(),
            ));
        }

        let padding_left = if filter_size.is_multiple_of(2) {
            ((filter_size / 2) as isize - filter_center - 1).max(0) as usize
        } else {
            ((filter_size / 2) as isize - filter_center).max(0) as usize
        };

        let padding_right = filter_size.saturating_sub(padding_left);

        let arena = make_arena_1d(input, padding_left, padding_right, self.border_mode)?;

        let c0 = unsafe { *kernel.get_unchecked(0) };

        for (x, dst) in output.chunks_exact_mut(4).enumerate() {
            unsafe {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = (*shifted_src.get_unchecked(0)).mul(c0);
                let mut k1 = (*shifted_src.get_unchecked(1)).mul(c0);
                let mut k2 = (*shifted_src.get_unchecked(2)).mul(c0);
                let mut k3 = (*shifted_src.get_unchecked(3)).mul(c0);

                for i in 1..filter_size {
                    let coeff = *kernel.get_unchecked(i);
                    k0 = fmla(*shifted_src.get_unchecked(i), coeff, k0);
                    k1 = fmla(*shifted_src.get_unchecked(i + 1), coeff, k1);
                    k2 = fmla(*shifted_src.get_unchecked(i + 2), coeff, k2);
                    k3 = fmla(*shifted_src.get_unchecked(i + 3), coeff, k3);
                }
                dst[0] = k0;
                dst[1] = k1;
                dst[2] = k2;
                dst[3] = k3;
            }
        }

        let p = output.chunks_exact_mut(4).len() * 4;
        let output = output.chunks_exact_mut(4).into_remainder();

        for (x, dst) in output.iter_mut().enumerate() {
            unsafe {
                let shifted_src = arena.get_unchecked(p + x..);

                let mut k0 = (*shifted_src.get_unchecked(0)).mul(c0);

                for i in 1..filter_size {
                    let coeff = *kernel.get_unchecked(i);
                    k0 = fmla(*shifted_src.get_unchecked(i), coeff, k0);
                }
                *dst = k0;
            }
        }

        Ok(())
    }
}
