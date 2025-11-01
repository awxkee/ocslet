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
use std::arch::aarch64::*;

pub(crate) struct NeonWavelet4TapsF32 {
    border_mode: BorderMode,
    low_pass: [f32; 4],
    high_pass: [f32; 4],
}

impl NeonWavelet4TapsF32 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f32; 4]) -> Self {
        let hp = low_pass_to_high_from_arr(wavelet);
        Self {
            border_mode,
            low_pass: [wavelet[0], wavelet[1], wavelet[2], wavelet[3]],
            high_pass: [hp[0], hp[1], hp[2], hp[3]],
        }
    }
}

impl DwtForwardExecutor<f32> for NeonWavelet4TapsF32 {
    fn execute_forward(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
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
            let h = vld1q_f32(self.low_pass.as_ptr());
            let g = vld1q_f32(self.high_pass.as_ptr());

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(4)
                .zip(details.chunks_exact_mut(4))
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = padded_input.get_unchecked(base0..);

                let xw01 = vld1q_f32(input0.as_ptr());
                let xw23 = vld1q_f32(input0.get_unchecked(4..).as_ptr());
                let xw3 = vld1q_f32(input0.get_unchecked(6..).as_ptr());

                let xw0 = xw01;
                let xw1 = vcombine_f32(vget_high_f32(xw01), vget_low_f32(xw23));
                let xw2 = xw23;

                let a0 = vmulq_f32(xw0, h);
                let d0 = vmulq_f32(xw0, g);

                let a1 = vmulq_f32(xw1, h);
                let d1 = vmulq_f32(xw1, g);

                let a2 = vmulq_f32(xw2, h);
                let d2 = vmulq_f32(xw2, g);

                let a3 = vmulq_f32(xw3, h);
                let d3 = vmulq_f32(xw3, g);

                let wa = vpaddq_f32(vpaddq_f32(a0, a1), vpaddq_f32(a2, a3));
                let wd = vpaddq_f32(vpaddq_f32(d0, d1), vpaddq_f32(d2, d3));

                vst1q_f32(approx.as_mut_ptr(), wa);
                vst1q_f32(detail.as_mut_ptr(), wd);

                processed += 4;
            }

            let approx = approx.chunks_exact_mut(4).into_remainder();
            let details = details.chunks_exact_mut(4).into_remainder();

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * (processed + i);

                let input = padded_input.get_unchecked(base..);

                let xw = vld1q_f32(input.as_ptr());

                let a = vmulq_f32(xw, h);
                let d = vmulq_f32(xw, g);

                let a0 = vpadds_f32(vadd_f32(vget_low_f32(a), vget_high_f32(a)));
                let d0 = vpadds_f32(vadd_f32(vget_low_f32(d), vget_high_f32(d)));

                *approx = a0;
                *detail = d0;
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f32> for NeonWavelet4TapsF32 {
    fn execute_inverse(
        &self,
        approx: &[f32],
        details: &[f32],
        output: &mut [f32],
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
            let safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;
            for i in 0..safe_start.min(safe_end) {
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

            let wh = vld1q_f32(self.low_pass.as_ptr());
            let wg = vld1q_f32(self.high_pass.as_ptr());

            let mut ui = safe_start;

            while ui + 4 < safe_end {
                let (h, g) = (
                    vld1q_f32(approx.get_unchecked(ui)),
                    vld1q_f32(details.get_unchecked(ui)),
                );
                let k = 2 * ui as isize - FILTER_OFFSET as isize;
                let part0 = output.get_unchecked_mut(k as usize..);
                let q0 = vld1q_f32(part0.as_ptr());
                let q1 = vld1q_f32(part0.get_unchecked(2..).as_ptr());
                let q2 = vld1q_f32(part0.get_unchecked(4..).as_ptr());
                let q3 = vld1q_f32(part0.get_unchecked(6..).as_ptr());
                let w0 = vfmaq_laneq_f32::<0>(vfmaq_laneq_f32::<0>(q0, wh, h), wg, g);
                let mut w1 = vfmaq_laneq_f32::<1>(vfmaq_laneq_f32::<1>(q1, wh, h), wg, g);
                w1 = vaddq_f32(w1, vcombine_f32(vget_high_f32(w0), vdup_n_f32(0.)));
                let mut w2 = vfmaq_laneq_f32::<2>(vfmaq_laneq_f32::<2>(q2, wh, h), wg, g);
                w2 = vaddq_f32(w2, vcombine_f32(vget_high_f32(w1), vdup_n_f32(0.)));
                let mut w3 = vfmaq_laneq_f32::<3>(vfmaq_laneq_f32::<3>(q3, wh, h), wg, g);
                w3 = vaddq_f32(w3, vcombine_f32(vget_high_f32(w2), vdup_n_f32(0.)));
                vst1q_f32(part0.as_mut_ptr(), w0);
                vst1q_f32(part0.get_unchecked_mut(2..).as_mut_ptr(), w1);
                vst1q_f32(part0.get_unchecked_mut(4..).as_mut_ptr(), w2);
                vst1q_f32(part0.get_unchecked_mut(6..).as_mut_ptr(), w3);
                ui += 4;
            }

            for i in ui..safe_end {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                let part = output.get_unchecked_mut(k as usize..);
                let q0 = vld1q_f32(part.as_ptr());
                let w0 = vfmaq_n_f32(vfmaq_n_f32(q0, wh, h), wg, g);
                vst1q_f32(part.as_mut_ptr(), w0);
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

impl IncompleteDwtExecutor<f32> for NeonWavelet4TapsF32 {
    fn filter_length(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db2_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db2 = NeonWavelet4TapsF32::new(
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

        const REFERENCE_APPROX: [f32; 10] = [
            2.68446737, 2.31078903, 5.11383217, 1.67303261, 0.53329969, 6.3061913, 7.60071774,
            2.95715649, 1.7599028, 2.10398276,
        ];
        const REFERENCE_DETAILS: [f32; 10] = [
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
                (REFERENCE_APPROX[i] - x).abs() < 1e-4,
                "approx difference expected to be < 1e-4, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-4,
                "details difference expected to be < 1e-4, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 4)];
        db2.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-4,
                "reconstructed difference expected to be < 1e-4, but values were ref {}, derived {}",
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
        let db4 = NeonWavelet4TapsF32::new(
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

        const REFERENCE_APPROX: [f32; 9] = [
            1.29427747, 2.31078903, 5.11383217, 1.67303261, 0.53329969, 6.3061913, 7.60071774,
            2.95715649, 1.29427747,
        ];
        const REFERENCE_DETAILS: [f32; 9] = [
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
                (REFERENCE_APPROX[i] - x).abs() < 1e-4,
                "approx difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-4,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 4)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-4,
                "reconstructed difference expected to be < 1e-4, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }
}
