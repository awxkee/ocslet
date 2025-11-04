/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::avx::util::_mm256_hsum_pd;
use crate::border_mode::BorderMode;
use crate::err::OscletError;
use crate::filter_padding::make_arena_1d;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr};
use crate::{DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor};
use std::arch::x86_64::*;

pub(crate) struct AvxWavelet10TapsF64 {
    border_mode: BorderMode,
    low_pass: [f64; 12],
    high_pass: [f64; 12],
}

impl AvxWavelet10TapsF64 {
    pub(crate) fn new(border_mode: BorderMode, w: &[f64; 10]) -> Self {
        let g = low_pass_to_high_from_arr(w);
        Self {
            border_mode,
            low_pass: [
                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], 0., 0.,
            ],
            high_pass: [
                g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], 0., 0.,
            ],
        }
    }
}

impl DwtForwardExecutor<f64> for AvxWavelet10TapsF64 {
    fn execute_forward(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_forward_impl(input, approx, details) }
    }
}

impl AvxWavelet10TapsF64 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_forward_impl(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        let half = dwt_length(input.len(), 10);

        if input.len() < 10 {
            return Err(OscletError::MinFilterSize(input.len(), 10));
        }

        if approx.len() != half {
            return Err(OscletError::ApproxDetailsSize(approx.len()));
        }
        if details.len() != half {
            return Err(OscletError::ApproxDetailsSize(details.len()));
        }

        const FILTER_SIZE: usize = 10;

        let whole_size = (2 * half + FILTER_SIZE - 2) - input.len();
        let left_pad = whole_size / 2;
        let right_pad = whole_size - left_pad;

        let padded_input = make_arena_1d(input, left_pad, right_pad, self.border_mode)?;

        unsafe {
            let h0 = _mm256_loadu_pd(self.low_pass.as_ptr());
            let g0 = _mm256_loadu_pd(self.high_pass.as_ptr());

            let h2 = _mm256_loadu_pd(self.low_pass.get_unchecked(4..).as_ptr());
            let g2 = _mm256_loadu_pd(self.high_pass.get_unchecked(4..).as_ptr());

            let h8 = _mm256_loadu_pd(self.low_pass.get_unchecked(8..).as_ptr());
            let g8 = _mm256_loadu_pd(self.high_pass.get_unchecked(8..).as_ptr());

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..);

                let x01 = _mm256_loadu_pd(input.as_ptr());
                let x45 = _mm256_loadu_pd(input.get_unchecked(4..).as_ptr());
                let x89 = _mm256_castpd128_pd256(_mm_loadu_pd(input.get_unchecked(8..).as_ptr()));

                let wa = _mm256_fmadd_pd(h8, x89, _mm256_fmadd_pd(x45, h2, _mm256_mul_pd(x01, h0)));
                let wd = _mm256_fmadd_pd(g8, x89, _mm256_fmadd_pd(x45, g2, _mm256_mul_pd(x01, g0)));

                let wa = _mm256_hsum_pd(wa);
                let wd = _mm256_hsum_pd(wd);

                _mm_storel_pd(approx as *mut f64, wa);
                _mm_storel_pd(detail as *mut f64, wd);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f64> for AvxWavelet10TapsF64 {
    fn execute_inverse(
        &self,
        approx: &[f64],
        details: &[f64],
        output: &mut [f64],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_inverse_impl(approx, details, output) }
    }
}

impl AvxWavelet10TapsF64 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_inverse_impl(
        &self,
        approx: &[f64],
        details: &[f64],
        output: &mut [f64],
    ) -> Result<(), OscletError> {
        if approx.len() != details.len() {
            return Err(OscletError::ApproxDetailsNotMatches(
                approx.len(),
                details.len(),
            ));
        }

        let rec_len = idwt_length(approx.len(), 10);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsTooSmall(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 8;
        const FILTER_LENGTH: usize = 10;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let mut safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;

            if safe_start < safe_end {
                for i in 0..safe_start {
                    let (h, g) = (
                        _mm_set1_pd(*approx.get_unchecked(i)),
                        _mm_set1_pd(*details.get_unchecked(i)),
                    );
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    for j in 0..10 {
                        let k = k + j as isize;
                        if k >= 0 && k < rec_len as isize {
                            let mut w = _mm_fmadd_sd(
                                _mm_set1_pd(self.high_pass[j]),
                                g,
                                _mm_load_sd(output.get_unchecked(k as usize) as *const f64),
                            );
                            w = _mm_fmadd_sd(_mm_set1_pd(self.low_pass[j]), h, w);
                            _mm_store_sd(output.get_unchecked_mut(k as usize) as *mut f64, w);
                        }
                    }
                }

                let h0 = _mm256_loadu_pd(self.low_pass.as_ptr());
                let g0 = _mm256_loadu_pd(self.high_pass.as_ptr());

                let h2 = _mm256_loadu_pd(self.low_pass.get_unchecked(4..).as_ptr());
                let g2 = _mm256_loadu_pd(self.high_pass.get_unchecked(4..).as_ptr());

                let h8 = _mm_loadu_pd(self.low_pass.get_unchecked(8..).as_ptr());
                let g8 = _mm_loadu_pd(self.high_pass.get_unchecked(8..).as_ptr());

                for i in safe_start..safe_end {
                    let (h, g) = (
                        _mm256_set1_pd(*approx.get_unchecked(i)),
                        _mm256_set1_pd(*details.get_unchecked(i)),
                    );
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);
                    let w0 = _mm256_loadu_pd(part.as_ptr());
                    let w2 = _mm256_loadu_pd(part.get_unchecked(4..).as_ptr());
                    let w8 = _mm_loadu_pd(part.get_unchecked(8..).as_ptr());

                    let q0 = _mm256_fmadd_pd(g0, g, _mm256_fmadd_pd(h0, h, w0));
                    let q4 = _mm256_fmadd_pd(g2, g, _mm256_fmadd_pd(h2, h, w2));
                    let q8 = _mm_fmadd_pd(
                        g8,
                        _mm256_castpd256_pd128(g),
                        _mm_fmadd_pd(h8, _mm256_castpd256_pd128(h), w8),
                    );

                    _mm256_storeu_pd(part.as_mut_ptr(), q0);
                    _mm256_storeu_pd(part.get_unchecked_mut(4..).as_mut_ptr(), q4);
                    _mm_storeu_pd(part.get_unchecked_mut(8..).as_mut_ptr(), q8);
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (
                    _mm_set1_pd(*approx.get_unchecked(i)),
                    _mm_set1_pd(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..10 {
                    let k = k + j as isize;
                    if k >= 0 && k < rec_len as isize {
                        let mut w = _mm_fmadd_sd(
                            _mm_set1_pd(self.high_pass[j]),
                            g,
                            _mm_load_sd(output.get_unchecked(k as usize) as *const f64),
                        );
                        w = _mm_fmadd_sd(_mm_set1_pd(self.low_pass[j]), h, w);
                        _mm_store_sd(output.get_unchecked_mut(k as usize) as *mut f64, w);
                    }
                }
            }
        }
        Ok(())
    }
}

impl IncompleteDwtExecutor<f64> for AvxWavelet10TapsF64 {
    fn filter_length(&self) -> usize {
        10
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db5_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db4 = AvxWavelet10TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db5
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 10);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 13] = [
            7.76308732, 4.31346037, 1.56478564, 2.0152442, 3.56281138, 4.58718134, 0.3541704,
            2.85175073, 5.65669193, 8.10768146, 1.19172576, 2.51812658, 1.90731395,
        ];
        const REFERENCE_DETAILS: [f64; 13] = [
            -1.63683102,
            0.4689461,
            -1.17579949,
            0.13929415,
            -0.43849194,
            -3.73867239,
            0.17971352,
            1.34570881,
            0.03313086,
            0.65749801,
            0.28165624,
            0.29616734,
            0.48934456,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-5,
                "approx difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-5,
                "details difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 10)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstructed difference expected to be < 1e-5, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db5_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db5 = AvxWavelet10TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db5
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 10);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db5.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 12] = [
            5.67889878, 7.9758377, 1.616079, 1.16256715, 3.56281138, 4.58718134, 0.3541704,
            2.85175073, 5.67889878, 7.9758377, 1.616079, 1.16256715,
        ];
        const REFERENCE_DETAILS: [f64; 12] = [
            -1.03271544,
            0.16927413,
            -1.06111809,
            0.12152867,
            -0.43849194,
            -3.73867239,
            0.17971352,
            1.34570881,
            -1.03271544,
            0.16927413,
            -1.06111809,
            0.12152867,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-5,
                "approx difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-5,
                "details difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 10)];
        db5.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstructed difference expected to be < 1e-5, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db8_even_big() {
        let data_length = 86;
        let mut input = vec![0.; data_length];
        for i in 0..data_length {
            input[i] = i as f64 / data_length as f64;
        }
        let db4 = AvxWavelet10TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db5
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 10);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 10)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstructed difference expected to be < 1e-5, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }
}
