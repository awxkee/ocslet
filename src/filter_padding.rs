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
use crate::border_mode::{BorderMode, reflect_index, reflect_index_101};
use crate::err::{OscletError, try_vec};
use num_traits::AsPrimitive;
use std::ops::Range;

pub(crate) fn make_arena_1d<T: Copy + Default + Clone + 'static>(
    data: &[T],
    pad_left: usize,
    pad_right: usize,
    border_mode: BorderMode,
) -> Result<Vec<T>, OscletError>
where
    f64: AsPrimitive<T>,
{
    let mut padded = try_vec![T::default(); pad_left + data.len() + pad_right];
    for (dst, src) in padded.iter_mut().skip(pad_left).zip(data.iter()) {
        *dst = *src;
    }

    let filling_ranges = [
        Range {
            start: 0,
            end: pad_left,
        },
        Range {
            start: padded.len() - pad_right,
            end: padded.len(),
        },
    ];

    for range in filling_ranges.iter() {
        match border_mode {
            BorderMode::Clamp => {
                let reshaped = &mut padded[range.start..range.end];
                for (idx, dst) in reshaped.iter_mut().enumerate() {
                    let item = (range.start as isize - pad_left as isize + idx as isize)
                        .min(data.len() as isize - 1)
                        .max(0) as usize;
                    unsafe {
                        *dst = *data.get_unchecked(item);
                    }
                }
            }
            BorderMode::Wrap => {
                let reshaped = &mut padded[range.start..range.end];
                for (idx, dst) in reshaped.iter_mut().enumerate() {
                    let item = (range.start as isize - pad_left as isize + idx as isize)
                        .rem_euclid(data.len() as isize) as usize;
                    unsafe {
                        *dst = *data.get_unchecked(item);
                    }
                }
            }
            BorderMode::Reflect => {
                let reshaped = &mut padded[range.start..range.end];
                for (idx, dst) in reshaped.iter_mut().enumerate() {
                    let item = reflect_index(
                        range.start as isize - pad_left as isize + idx as isize,
                        data.len() as isize,
                    );
                    unsafe {
                        *dst = *data.get_unchecked(item);
                    }
                }
            }
            BorderMode::Reflect101 => {
                let reshaped = &mut padded[range.start..range.end];
                for (idx, dst) in reshaped.iter_mut().enumerate() {
                    let item = reflect_index_101(
                        range.start as isize - pad_left as isize + idx as isize,
                        data.len() as isize,
                    );
                    unsafe {
                        *dst = *data.get_unchecked(item);
                    }
                }
            }
            BorderMode::Zeros => {
                let reshaped = &mut padded[range.start..range.end];
                for (idx, dst) in reshaped.iter_mut().enumerate() {
                    let idx = range.start as isize - pad_left as isize + idx as isize;
                    unsafe {
                        if idx < 0 || idx >= data.len() as isize - 1 {
                            *dst = 0f64.as_();
                        } else {
                            *dst = *data.get_unchecked(idx as usize);
                        }
                    }
                }
            }
        }
    }

    Ok(padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padding() {
        let data = [1, 2, 3, 4, 5];

        let arena1 = make_arena_1d::<u8>(&data, 3, 3, BorderMode::Clamp).unwrap();
        assert_eq!(arena1[0], 1);
        assert_eq!(arena1[8], 5);

        let arena2 = make_arena_1d::<u8>(&data, 2, 2, BorderMode::Wrap).unwrap();
        assert_eq!(arena2[1], 5);
        assert_eq!(arena2[7], 1);

        let arena3 = make_arena_1d::<u8>(&data, 7, 7, BorderMode::Reflect).unwrap();
        assert_eq!(arena3[0], 2);
        assert_eq!(arena3[1], 1);
        assert_eq!(arena3[8], 2);

        let arena4 = make_arena_1d::<u8>(&data, 2, 2, BorderMode::Reflect101).unwrap();
        assert_eq!(arena4[0], 3);
        assert_eq!(arena4[1], 2);
        assert_eq!(arena4[7], 4);
        assert_eq!(arena4[8], 3);

        println!("{}", reflect_index_101(15, 1));
    }
}
