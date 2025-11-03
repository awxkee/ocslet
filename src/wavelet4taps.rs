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
use crate::border_mode::BorderMode;
use crate::err::OscletError;
use crate::filter_padding::make_arena_1d;
use crate::mla::fmla;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr};
use crate::{DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor};
use num_traits::{AsPrimitive, MulAdd};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

pub(crate) struct Wavelet4Taps<T> {
    phantom_data: PhantomData<T>,
    border_mode: BorderMode,
    low_pass: [T; 4],
    high_pass: [T; 4],
}

impl<T: Copy + 'static + Debug + Default + Mul<T, Output = T>> Wavelet4Taps<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[T; 4]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
            phantom_data: PhantomData,
        }
    }
}

impl<T: Copy + 'static + MulAdd<T, Output = T> + Add<T, Output = T> + Mul<T, Output = T> + Default>
    DwtForwardExecutor<T> for Wavelet4Taps<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_forward(
        &self,
        input: &[T],
        approx: &mut [T],
        details: &mut [T],
    ) -> Result<(), OscletError> {
        let half = dwt_length(input.len(), 4);

        if input.len() < 4 {
            return Err(OscletError::MinFilterSize(input.len(), 4));
        }

        if approx.len() != half {
            return Err(OscletError::ApproxDetailsSize(approx.len()));
        }
        if details.len() != half {
            return Err(OscletError::ApproxDetailsSize(details.len()));
        }

        let padded_input = make_arena_1d(
            input,
            2,
            if !input.len().is_multiple_of(2) { 3 } else { 2 },
            self.border_mode,
        )?;

        unsafe {
            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..);

                let x0 = input.get_unchecked(0);
                let x1 = input.get_unchecked(1);
                let x2 = input.get_unchecked(2);
                let x3 = input.get_unchecked(3);

                a = fmla(self.low_pass[0], *x0, a);
                d = fmla(self.high_pass[0], *x0, d);

                a = fmla(self.low_pass[1], *x1, a);
                d = fmla(self.high_pass[1], *x1, d);

                a = fmla(self.low_pass[2], *x2, a);
                d = fmla(self.high_pass[2], *x2, d);

                a = fmla(self.low_pass[3], *x3, a);
                d = fmla(self.high_pass[3], *x3, d);

                *approx = a;
                *detail = d;
            }
        }
        Ok(())
    }
}

impl<T: Copy + 'static + MulAdd<T, Output = T> + Add<T, Output = T> + Mul<T, Output = T> + Default>
    DwtInverseExecutor<T> for Wavelet4Taps<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_inverse(
        &self,
        approx: &[T],
        details: &[T],
        output: &mut [T],
    ) -> Result<(), OscletError> {
        if approx.len() != details.len() {
            return Err(OscletError::ApproxDetailsNotMatches(
                approx.len(),
                details.len(),
            ));
        }

        let rec_len = idwt_length(approx.len(), 4);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsTooSmall(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 2;
        const FILTER_LENGTH: usize = 4;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let mut safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;

            if safe_start < safe_end {
                for i in 0..safe_start {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    for j in 0..4 {
                        let k = k + j as isize;
                        if k >= 0 && k < rec_len as isize {
                            *output.get_unchecked_mut(k as usize) = fmla(
                                self.low_pass[j],
                                h,
                                fmla(self.high_pass[j], g, *output.get_unchecked(k as usize)),
                            );
                        }
                    }
                }

                for i in safe_start..safe_end {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);
                    *part.get_unchecked_mut(0) = fmla(
                        self.low_pass[0],
                        h,
                        fmla(self.high_pass[0], g, *part.get_unchecked(0)),
                    );
                    *part.get_unchecked_mut(1) = fmla(
                        self.low_pass[1],
                        h,
                        fmla(self.high_pass[1], g, *part.get_unchecked(1)),
                    );
                    *part.get_unchecked_mut(2) = fmla(
                        self.low_pass[2],
                        h,
                        fmla(self.high_pass[2], g, *part.get_unchecked(2)),
                    );
                    *part.get_unchecked_mut(3) = fmla(
                        self.low_pass[3],
                        h,
                        fmla(self.high_pass[3], g, *part.get_unchecked(3)),
                    );
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..4 {
                    let k = k + j as isize;
                    if k >= 0 && k < rec_len as isize {
                        *output.get_unchecked_mut(k as usize) = fmla(
                            self.low_pass[j],
                            h,
                            fmla(self.high_pass[j], g, *output.get_unchecked(k as usize)),
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

impl<
    T: Copy
        + 'static
        + MulAdd<T, Output = T>
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Default
        + Send
        + Sync,
> IncompleteDwtExecutor<T> for Wavelet4Taps<T>
where
    f64: AsPrimitive<T>,
{
    fn filter_length(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symlets::SymletFamily;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db2_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db2 = Wavelet4Taps::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db2
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 4);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db2.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 10] = [
            2.68446737, 2.31078903, 5.11383217, 1.67303261, 0.53329969, 6.3061913, 7.60071774,
            2.95715649, 1.7599028, 2.10398276,
        ];
        const REFERENCE_DETAILS: [f64; 10] = [
            -8.58001572e-01,
            1.00000000e-16,
            -9.47343455e-02,
            -9.65925826e-01,
            -1.35576367e+00,
            -2.85084151e+00,
            2.31500342e+00,
            -1.01700947e+00,
            1.25223606e+00,
            -3.23523806e-01,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-7,
                "approx difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        println!("{:?}", approx);
        println!("{:?}", details);

        let mut reconstructed = vec![
            0.0;
            if input.len() % 2 != 0 {
                input.len() + 1
            } else {
                input.len()
            }
        ];
        db2.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-7,
                "reconstructed difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db2_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = Wavelet4Taps::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db2
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 4);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 9] = [
            1.29427747, 2.31078903, 5.11383217, 1.67303261, 0.53329969, 6.3061913, 7.60071774,
            2.95715649, 1.29427747,
        ];
        const REFERENCE_DETAILS: [f64; 9] = [
            -4.85501312e-01,
            1.00000000e-16,
            -9.47343455e-02,
            -9.65925826e-01,
            -1.35576367e+00,
            -2.85084151e+00,
            2.31500342e+00,
            -1.01700947e+00,
            -4.85501312e-01,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-7,
                "approx difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![
            0.0;
            if input.len() % 2 != 0 {
                input.len() + 1
            } else {
                input.len()
            }
        ];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-7,
                "reconstructed difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_sym2_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = Wavelet4Taps::new(
            BorderMode::Wrap,
            SymletFamily::Sym2
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 4);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 9] = [
            1.29427747, 2.31078903, 5.11383217, 1.67303261, 0.53329969, 6.3061913, 7.60071774,
            2.95715649, 1.29427747,
        ];
        const REFERENCE_DETAILS: [f64; 9] = [
            -4.85501312e-01,
            1.00000000e-16,
            -9.47343455e-02,
            -9.65925826e-01,
            -1.35576367e+00,
            -2.85084151e+00,
            2.31500342e+00,
            -1.01700947e+00,
            -4.85501312e-01,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-7,
                "approx difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![
            0.0;
            if input.len() % 2 != 0 {
                input.len() + 1
            } else {
                input.len()
            }
        ];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-7,
                "reconstructed difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }
}
