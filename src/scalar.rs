use core::{
    fmt::Display,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use group::ff::{Field, PrimeField};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::bindings::{self, blst_fr};

/// Constant representing the modulus in little-endian u64 limbs.
///
/// q = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
pub const MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Scalar(pub(crate) blst_fr);

impl Display for Scalar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut bytes = self.to_repr();
        bytes.reverse();

        write!(f, "Scalar(0x")?;
        for b in bytes {
            write!(f, "{:02x}", b)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl Add for Scalar {
    type Output = Self;

    #[allow(clippy::op_ref)]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl Add<&Scalar> for Scalar {
    type Output = Self;

    fn add(mut self, rhs: &Scalar) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign<&Scalar> for Scalar {
    fn add_assign(&mut self, rhs: &Scalar) {
        // Safety: It is safe to call with `out` being one of the parameters.
        unsafe { bindings::blst_fr_add(&mut self.0, &self.0, &rhs.0) };
    }
}

impl AddAssign for Scalar {
    fn add_assign(&mut self, rhs: Scalar) {
        *self += &rhs;
    }
}

impl Sum for Scalar {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Scalar::ZERO, |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Scalar> for Scalar {
    fn sum<I: Iterator<Item = &'a Scalar>>(iter: I) -> Self {
        iter.fold(Scalar::ZERO, |acc, x| acc + x)
    }
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        // Safety: It is safe to call with `out` being one of the parameters.
        unsafe { bindings::blst_fr_cneg(&mut self.0, &self.0, true) };
        self
    }
}

impl Neg for &Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        -(*self)
    }
}

impl Sub for Scalar {
    type Output = Self;

    #[allow(clippy::op_ref)]
    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl Sub<&Scalar> for Scalar {
    type Output = Self;

    fn sub(self, rhs: &Scalar) -> Self::Output {
        self + -rhs
    }
}

impl SubAssign<&Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: &Scalar) {
        *self += -rhs;
    }
}

impl SubAssign for Scalar {
    fn sub_assign(&mut self, rhs: Scalar) {
        *self += -rhs;
    }
}

impl Mul for Scalar {
    type Output = Self;

    #[allow(clippy::op_ref)]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Scalar> for Scalar {
    type Output = Self;

    fn mul(mut self, rhs: &Scalar) -> Self::Output {
        self *= rhs;
        self
    }
}

impl MulAssign<&Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: &Scalar) {
        // Safety: It is safe to call with `out` being one of the parameters.
        unsafe { bindings::blst_fr_mul(&mut self.0, &self.0, &rhs.0) };
    }
}

impl MulAssign for Scalar {
    fn mul_assign(&mut self, rhs: Scalar) {
        *self *= &rhs;
    }
}

impl Product for Scalar {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Scalar::ONE, |acc, x| acc * x)
    }
}

impl<'a> Product<&'a Scalar> for Scalar {
    fn product<I: Iterator<Item = &'a Scalar>>(iter: I) -> Self {
        iter.fold(Scalar::ONE, |acc, x| acc * x)
    }
}

impl ConstantTimeEq for Scalar {
    fn ct_eq(&self, other: &Self) -> subtle::Choice {
        self.0.l.ct_eq(&other.0.l)
    }
}

impl ConditionallySelectable for Scalar {
    fn conditional_select(a: &Self, b: &Self, choice: subtle::Choice) -> Self {
        Scalar(blst_fr {
            l: [
                u64::conditional_select(&a.0.l[0], &b.0.l[0], choice),
                u64::conditional_select(&a.0.l[1], &b.0.l[1], choice),
                u64::conditional_select(&a.0.l[2], &b.0.l[2], choice),
                u64::conditional_select(&a.0.l[3], &b.0.l[3], choice),
            ],
        })
    }
}

impl Ord for Scalar {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        for i in (0..4).rev() {
            if self.0.l[i] < other.0.l[i] {
                return core::cmp::Ordering::Less;
            } else if self.0.l[i] > other.0.l[i] {
                return core::cmp::Ordering::Greater;
            }
        }
        core::cmp::Ordering::Equal
    }
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl From<u64> for Scalar {
    fn from(value: u64) -> Self {
        let mut repr = [0u8; 32];
        repr[..8].copy_from_slice(&value.to_le_bytes());
        Scalar::from_repr(repr).expect("a value within [0..2^64] is a valid scalar")
    }
}

impl Field for Scalar {
    const ZERO: Self = Scalar(blst_fr { l: [0, 0, 0, 0] });

    const ONE: Self = Scalar(blst_fr {
        l: [
            0x0000_0001_ffff_fffe,
            0x5884_b7fa_0003_4802,
            0x998c_4fef_ecbc_4ff5,
            0x1824_b159_acc5_056f,
        ],
    });

    fn try_from_rng<R: rand_core::TryRng + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        loop {
            let mut scalar = bindings::blst_fr { l: [0, 0, 0, 0] };
            // Safety:
            // - Both types have the same size, and are `repr(C)`.
            // - `blst_fr` has stricter alignment (`u64`) than `blst_scalar` (`u8`).
            // - `scalar` is not used until the call to `blst_fr_from_scalar`.
            let scalar_as_bytes = unsafe {
                &mut *(&mut scalar as *mut bindings::blst_fr as *mut bindings::blst_scalar)
            };
            rng.try_fill_bytes(&mut scalar_as_bytes.b)?;

            // Mask away the unused most-significant bits.
            scalar_as_bytes.b[31] &= 0xff >> (256 - Scalar::NUM_BITS as usize);

            if unsafe { bindings::blst_scalar_fr_check(scalar_as_bytes) } {
                // Safety: The function allows for `out` and `scalar` to be the same pointer.
                unsafe { bindings::blst_fr_from_scalar(&mut scalar, scalar_as_bytes) };
                let _ = scalar_as_bytes;
                // Safety: aligned, valid, and unique pointer.
                return Ok(Scalar(scalar));
            }
        }
    }

    fn square(&self) -> Self {
        let mut out = *self;
        // Safety: Inputs are valid, and respect the ffi contract.
        unsafe {
            bindings::blst_fr_sqr(&mut (out.0), &self.0);
        }
        out
    }

    fn double(&self) -> Self {
        let mut out = *self;
        out += self;
        out
    }

    fn invert(&self) -> subtle::CtOption<Self> {
        let mut out = *self;
        // Safety: Inputs are valid, and respect the ffi contract.
        unsafe { bindings::blst_fr_eucl_inverse(&mut out.0, &self.0) };
        subtle::CtOption::new(out, !self.is_zero())
    }

    fn sqrt(&self) -> CtOption<Self> {
        // (t - 1) // 2 = 6104339283789297388802252303364915521546564123189034618274734669823
        group::ff::helpers::sqrt_tonelli_shanks(
            self,
            [
                0x7fff_2dff_7fff_ffff,
                0x04d0_ec02_a9de_d201,
                0x94ce_bea4_199c_ec04,
                0x0000_0000_39f6_d3a9,
            ],
        )
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (subtle::Choice, Self) {
        group::ff::helpers::sqrt_ratio_generic(num, div)
    }
}

impl PrimeField for Scalar {
    type Repr = [u8; 32];

    fn from_repr(repr: Self::Repr) -> subtle::CtOption<Self> {
        let mut scalar = bindings::blst_scalar { b: repr };
        let is_valid = unsafe { bindings::blst_scalar_fr_check(&scalar) };
        let out_ptr: *mut blst_fr = &mut scalar as *mut bindings::blst_scalar as *mut blst_fr;
        // Safety: both types have the same size, and are `repr(C)`. The out pointer can be
        // the same as the input pointer.
        unsafe { bindings::blst_fr_from_scalar(out_ptr, &scalar) };
        // Safety: aligned, valid, and unique pointer.
        let out = Scalar(unsafe { *out_ptr });
        CtOption::new(
            out,
            // Safety: scalar is valid, and respects the ffi contract.
            Choice::from(is_valid as u8),
        )
    }

    fn to_repr(&self) -> Self::Repr {
        let mut out = bindings::blst_scalar { b: [0u8; 32] };
        // Safety: Inputs are valid, and respect the ffi contract.
        unsafe { bindings::blst_scalar_from_fr(&mut out, &self.0) };
        out.b
    }

    fn is_odd(&self) -> subtle::Choice {
        Choice::from(self.to_repr()[0] & 1)
    }

    const MODULUS: &'static str =
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";

    const NUM_BITS: u32 = 255;

    const CAPACITY: u32 = Self::NUM_BITS - 1;

    const TWO_INV: Self = Scalar(blst_fr {
        l: [
            0x0000_0000_ffff_ffff,
            0xac42_5bfd_0001_a401,
            0xccc6_27f7_f65e_27fa,
            0x0c12_58ac_d662_82b7,
        ],
    });

    const MULTIPLICATIVE_GENERATOR: Self = Scalar(blst_fr {
        l: [
            0x0000_000e_ffff_fff1,
            0x17e3_63d3_0018_9c0f,
            0xff9c_5787_6f84_57b0,
            0x3513_3220_8fc5_a8c4,
        ],
    });

    const S: u32 = 32;

    const ROOT_OF_UNITY: Self = Scalar(blst_fr {
        l: [
            0xb9b5_8d8c_5f0e_466a,
            0x5b1b_4c80_1819_d7ec,
            0x0af5_3ae3_52a3_1e64,
            0x5bf3_adda_19e9_b27b,
        ],
    });

    const ROOT_OF_UNITY_INV: Self = Scalar(blst_fr {
        l: [
            0x4256_481a_dcf3_219a,
            0x45f3_7b7f_96b6_cad3,
            0xf9c3_f1d7_5f7a_3b27,
            0x2d2f_c049_658a_fd43,
        ],
    });

    const DELTA: Self = Scalar(blst_fr {
        l: [
            0x70e3_10d3_d146_f96a,
            0x4b64_c089_19e2_99e6,
            0x51e1_1418_6a8b_970d,
            0x6185_d066_27c0_67cb,
        ],
    });
}

#[cfg(test)]
mod tests {
    use getrandom::SysRng;

    use super::*;

    /// INV = -(q^{-1} mod 2^64) mod 2^64
    const INV: u64 = 0xfffffffeffffffff;

    // Largest valid scalar in little-endian Montgomery form.
    const LARGEST: Scalar = Scalar(blst_fr {
        l: [
            0xffffffff00000000,
            0x53bda402fffe5bfe,
            0x3339d80809a1d805,
            0x73eda753299d7d48,
        ],
    });

    // Little-endian non-Montgomery form not reduced mod p.
    const MODULUS_REPR: [u8; 32] = [
        0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x5b, 0xfe, 0xff, 0x02, 0xa4, 0xbd,
        0x53, 0x05, 0xd8, 0xa1, 0x09, 0x08, 0xd8, 0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29, 0x53, 0xa7,
        0xed, 0x73,
    ];

    // `R = 2^256 mod q` in little-endian Montgomery form which is equivalent to 1 in little-endian
    // non-Montgomery form.
    //
    // sage> mod(2^256, 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001)
    // sage> 0x1824b159acc5056f998c4fefecbc4ff55884b7fa0003480200000001fffffffe
    const R: Scalar = Scalar(blst_fr {
        l: [
            0x0000_0001_ffff_fffe,
            0x5884_b7fa_0003_4802,
            0x998c_4fef_ecbc_4ff5,
            0x1824_b159_acc5_056f,
        ],
    });

    // `R^2 = 2^512 mod q` in little-endian Montgomery form.
    //
    // sage> mod(2^512, 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001)
    // sage> 0x748d9d99f59ff1105d314967254398f2b6cedcb87925c23c999e990f3f29c6d
    const R2: Scalar = Scalar(blst_fr {
        l: [
            0xc999_e990_f3f2_9c6d,
            0x2b6c_edcb_8792_5c23,
            0x05d3_1496_7254_398f,
            0x0748_d9d9_9f59_ff11,
        ],
    });

    #[test]
    fn inverse() {
        // Compute -(q^{-1} mod 2^64) mod 2^64 by exponentiating
        // by totient(2**64) - 1
        let mut inv = 1u64;
        for _ in 0..63 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(MODULUS[0]);
        }
        inv = inv.wrapping_neg();

        assert_eq!(inv, INV);

        assert_eq!(Scalar::ZERO.invert().is_none().unwrap_u8(), 1);

        let one = Scalar::ONE;

        for i in 0..1000 {
            // Ensure that a * a^-1 = 1
            let mut a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let ainv = a.invert().unwrap();
            a.mul_assign(&ainv);
            assert_eq!(a, one, "round {}", i);
        }
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn display() {
        use alloc::format;

        assert_eq!(
            format!("{}", Scalar::ZERO),
            "Scalar(0x0000000000000000000000000000000000000000000000000000000000000000)"
        );
        assert_eq!(
            format!("{}", Scalar::ONE),
            "Scalar(0x0000000000000000000000000000000000000000000000000000000000000001)"
        );
        assert_eq!(
            format!("{}", R2),
            "Scalar(0x1824b159acc5056f998c4fefecbc4ff55884b7fa0003480200000001fffffffe)"
        );
    }

    #[test]
    fn equality() {
        assert_eq!(Scalar::ZERO, Scalar::ZERO);
        assert_eq!(Scalar::ONE, Scalar::ONE);

        assert_ne!(Scalar::ZERO, Scalar::ONE);
        assert_ne!(Scalar::ONE, R2);
    }

    #[test]
    fn to_repr() {
        assert_eq!(Scalar::ZERO.to_repr(), [0; 32]);

        assert_eq!(Scalar::ONE.to_repr(), {
            let mut arr = [0; 32];
            arr[0] = 1;
            arr
        });

        assert_eq!(
            R2.to_repr(),
            [
                254, 255, 255, 255, 1, 0, 0, 0, 2, 72, 3, 0, 250, 183, 132, 88, 245, 79, 188, 236,
                239, 79, 140, 153, 111, 5, 197, 172, 89, 177, 36, 24
            ]
        );

        assert_eq!(
            (-Scalar::ONE).to_repr(),
            [
                0, 0, 0, 0, 255, 255, 255, 255, 254, 91, 254, 255, 2, 164, 189, 83, 5, 216, 161, 9,
                8, 216, 57, 51, 72, 125, 157, 41, 83, 167, 237, 115
            ]
        );
    }

    #[test]
    fn from_repr() {
        assert_eq!(
            Scalar::from_repr([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0
            ])
            .unwrap(),
            Scalar::ZERO
        );

        assert_eq!(
            Scalar::from_repr([
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0
            ])
            .unwrap(),
            Scalar::ONE
        );

        assert_eq!(
            Scalar::from_repr([
                254, 255, 255, 255, 1, 0, 0, 0, 2, 72, 3, 0, 250, 183, 132, 88, 245, 79, 188, 236,
                239, 79, 140, 153, 111, 5, 197, 172, 89, 177, 36, 24
            ])
            .unwrap(),
            R2,
        );

        // -1 should work
        assert!(bool::from(
            Scalar::from_repr([
                0, 0, 0, 0, 255, 255, 255, 255, 254, 91, 254, 255, 2, 164, 189, 83, 5, 216, 161, 9,
                8, 216, 57, 51, 72, 125, 157, 41, 83, 167, 237, 115
            ])
            .is_some()
        ));

        // modulus is invalid
        assert!(bool::from(Scalar::from_repr(MODULUS_REPR).is_none()));

        // Anything larger than the modulus is invalid
        assert!(bool::from(
            Scalar::from_repr([
                2, 0, 0, 0, 255, 255, 255, 255, 254, 91, 254, 255, 2, 164, 189, 83, 5, 216, 161, 9,
                8, 216, 57, 51, 72, 125, 157, 41, 83, 167, 237, 115
            ])
            .is_none()
        ));
        assert!(bool::from(
            Scalar::from_repr([
                1, 0, 0, 0, 255, 255, 255, 255, 254, 91, 254, 255, 2, 164, 189, 83, 5, 216, 161, 9,
                8, 216, 58, 51, 72, 125, 157, 41, 83, 167, 237, 115
            ])
            .is_none()
        ));
        assert!(bool::from(
            Scalar::from_repr([
                1, 0, 0, 0, 255, 255, 255, 255, 254, 91, 254, 255, 2, 164, 189, 83, 5, 216, 161, 9,
                8, 216, 57, 51, 72, 125, 157, 41, 83, 167, 237, 116
            ])
            .is_none()
        ));
    }

    #[test]
    fn repr_roundtrip() {
        let a = Scalar::from(1);
        let mut expected_bytes = [0u8; 32];
        expected_bytes[0] = 1;
        assert_eq!(a, Scalar::from_repr(a.to_repr()).unwrap());
        assert_eq!(a.to_repr(), expected_bytes);
        assert_eq!(a, Scalar::from_repr(expected_bytes).unwrap());

        let a = Scalar::from(12);
        let mut expected_bytes = [0u8; 32];
        expected_bytes[0] = 12;
        assert_eq!(a, Scalar::from_repr(a.to_repr()).unwrap());
        assert_eq!(a.to_repr(), expected_bytes);
        assert_eq!(a, Scalar::from_repr(expected_bytes).unwrap());
    }

    #[test]
    fn zero() {
        assert_eq!(Scalar::ZERO, -&Scalar::ZERO);
        assert_eq!(Scalar::ZERO, Scalar::ZERO + Scalar::ZERO);
        assert_eq!(Scalar::ZERO, Scalar::ZERO - Scalar::ZERO);
        assert_eq!(Scalar::ZERO, Scalar::ZERO * Scalar::ZERO);
    }

    #[test]
    fn addition() {
        let mut tmp = LARGEST;
        tmp += &LARGEST;

        assert_eq!(
            tmp,
            Scalar(blst_fr {
                l: [
                    0xfffffffeffffffff,
                    0x53bda402fffe5bfe,
                    0x3339d80809a1d805,
                    0x73eda753299d7d48
                ]
            })
        );

        let mut tmp = LARGEST;
        tmp += &Scalar(blst_fr { l: [1, 0, 0, 0] });

        assert_eq!(tmp, Scalar::ZERO);

        {
            // Random number
            let mut tmp = Scalar(blst_fr {
                l: [
                    0x437ce7616d580765,
                    0xd42d1ccb29d1235b,
                    0xed8f753821bd1423,
                    0x4eede1c9c89528ca,
                ],
            });
            // assert!(tmp.is_valid());
            // Test that adding zero has no effect.
            tmp.add_assign(&Scalar(blst_fr { l: [0, 0, 0, 0] }));
            assert_eq!(
                tmp,
                Scalar(blst_fr {
                    l: [
                        0x437ce7616d580765,
                        0xd42d1ccb29d1235b,
                        0xed8f753821bd1423,
                        0x4eede1c9c89528ca
                    ]
                })
            );
            // Add one and test for the result.
            tmp.add_assign(&Scalar(blst_fr { l: [1, 0, 0, 0] }));
            assert_eq!(
                tmp,
                Scalar(blst_fr {
                    l: [
                        0x437ce7616d580766,
                        0xd42d1ccb29d1235b,
                        0xed8f753821bd1423,
                        0x4eede1c9c89528ca
                    ]
                })
            );
            // Add another random number that exercises the reduction.
            tmp.add_assign(&Scalar(blst_fr {
                l: [
                    0x946f435944f7dc79,
                    0xb55e7ee6533a9b9b,
                    0x1e43b84c2f6194ca,
                    0x58717ab525463496,
                ],
            }));
            assert_eq!(
                tmp,
                Scalar(blst_fr {
                    l: [
                        0xd7ec2abbb24fe3de,
                        0x35cdf7ae7d0d62f7,
                        0xd899557c477cd0e9,
                        0x3371b52bc43de018
                    ]
                })
            );
            // Add one to (r - 1) and test for the result.
            tmp = Scalar(blst_fr {
                l: [
                    0xffffffff00000000,
                    0x53bda402fffe5bfe,
                    0x3339d80809a1d805,
                    0x73eda753299d7d48,
                ],
            });
            tmp.add_assign(&Scalar(blst_fr { l: [1, 0, 0, 0] }));
            assert!(bool::from(tmp.is_zero()));
            // Add such that the result is r - 1
            tmp = Scalar(blst_fr {
                l: [
                    0xade5adacdccb6190,
                    0xaa21ee0f27db3ccd,
                    0x2550f4704ae39086,
                    0x591d1902e7c5ba27,
                ],
            });
            tmp.add_assign(&Scalar(blst_fr {
                l: [
                    0x521a525223349e70,
                    0xa99bb5f3d8231f31,
                    0xde8e397bebe477e,
                    0x1ad08e5041d7c321,
                ],
            }));
            assert_eq!(
                tmp,
                Scalar(blst_fr {
                    l: [
                        0xffffffff00000000,
                        0x53bda402fffe5bfe,
                        0x3339d80809a1d805,
                        0x73eda753299d7d48
                    ]
                })
            );
            // Add one to the result
            tmp.add_assign(&Scalar(blst_fr { l: [1, 0, 0, 0] }));
            assert!(bool::from(tmp.is_zero()));
        }

        // Test associativity

        for i in 0..1000 {
            // Generate a, b, c and ensure (a + b) + c == a + (b + c).
            let a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let b = Scalar::try_from_rng(&mut SysRng).unwrap();
            let c = Scalar::try_from_rng(&mut SysRng).unwrap();

            let mut tmp1 = a;
            tmp1.add_assign(&b);
            tmp1.add_assign(&c);

            let mut tmp2 = b;
            tmp2.add_assign(&c);
            tmp2.add_assign(&a);

            // assert!(tmp1.is_valid());
            // assert!(tmp2.is_valid());
            assert_eq!(tmp1, tmp2, "round {}", i);
        }
    }

    #[test]
    fn negation() {
        let tmp = -&LARGEST;

        assert_eq!(tmp, Scalar(blst_fr { l: [1, 0, 0, 0] }));

        let tmp = -&Scalar::ZERO;
        assert_eq!(tmp, Scalar::ZERO);
        let tmp = -&Scalar(blst_fr { l: [1, 0, 0, 0] });
        assert_eq!(tmp, LARGEST);

        {
            let mut a = Scalar::ZERO;
            a = -a;

            assert!(bool::from(a.is_zero()));
        }

        for _ in 0..1000 {
            // Ensure (a - (-a)) = 0.
            let mut a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let mut b = a;
            b = -b;
            a += &b;

            assert!(bool::from(a.is_zero()));
        }
    }

    #[test]
    fn subtraction() {
        let mut tmp = LARGEST;
        tmp -= &LARGEST;

        assert_eq!(tmp, Scalar::ZERO);

        let mut tmp = Scalar::ZERO;
        tmp -= &LARGEST;

        let mut tmp2 = Scalar(blst_fr { l: MODULUS });
        tmp2 -= &LARGEST;

        assert_eq!(tmp, tmp2);

        for _ in 0..1000 {
            // Ensure that (a - b) + (b - a) = 0.
            let a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let b = Scalar::try_from_rng(&mut SysRng).unwrap();

            let mut tmp1 = a;
            tmp1.sub_assign(&b);

            let mut tmp2 = b;
            tmp2.sub_assign(&a);

            tmp1.add_assign(&tmp2);
            assert!(bool::from(tmp1.is_zero()));
        }
    }

    #[test]
    fn multiplication() {
        let mut tmp = Scalar(blst_fr {
            l: [
                0x6b7e9b8faeefc81a,
                0xe30a8463f348ba42,
                0xeff3cb67a8279c9c,
                0x3d303651bd7c774d,
            ],
        });
        tmp *= &Scalar(blst_fr {
            l: [
                0x13ae28e3bc35ebeb,
                0xa10f4488075cae2c,
                0x8160e95a853c3b5d,
                0x5ae3f03b561a841d,
            ],
        });
        assert!(
            tmp == Scalar(blst_fr {
                l: [
                    0x23717213ce710f71,
                    0xdbee1fe53a16e1af,
                    0xf565d3e1c2a48000,
                    0x4426507ee75df9d7
                ]
            })
        );

        for _ in 0..10000 {
            // Ensure that (a * b) * c = a * (b * c)
            let a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let b = Scalar::try_from_rng(&mut SysRng).unwrap();
            let c = Scalar::try_from_rng(&mut SysRng).unwrap();

            let mut tmp1 = a;
            tmp1 *= &b;
            tmp1 *= &c;

            let mut tmp2 = b;
            tmp2 *= &c;
            tmp2 *= &a;

            assert_eq!(tmp1, tmp2);
        }

        for _ in 0..10000 {
            // Ensure that r * (a + b + c) = r*a + r*b + r*c

            let r = Scalar::try_from_rng(&mut SysRng).unwrap();
            let mut a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let mut b = Scalar::try_from_rng(&mut SysRng).unwrap();
            let mut c = Scalar::try_from_rng(&mut SysRng).unwrap();

            let mut tmp1 = a;
            tmp1 += &b;
            tmp1 += &c;
            tmp1 *= &r;

            a *= &r;
            b *= &r;
            c *= &r;

            a += &b;
            a += &c;

            assert_eq!(tmp1, a);
        }
    }

    #[test]
    fn inverse_is_pow() {
        let q_minus_2 = [
            0xfffffffeffffffff,
            0x53bda402fffe5bfe,
            0x3339d80809a1d805,
            0x73eda753299d7d48,
        ];

        let mut r1 = R;
        let mut r2 = r1;

        for _ in 0..100 {
            r1 = r1.invert().unwrap();
            r2 = r2.pow_vartime(q_minus_2);

            assert_eq!(r1, r2);
            // Add R so we check something different next time around
            r1 += R;
            r2 = r1;
        }
    }

    #[test]
    fn sqrt() {
        {
            assert_eq!(Scalar::ZERO.sqrt().unwrap(), Scalar::ZERO);
            assert_eq!(Scalar::ONE.sqrt().unwrap(), Scalar::ONE);
        }

        let mut square = Scalar(blst_fr {
            l: [
                0x46cd85a5f273077e,
                0x1d30c47dd68fc735,
                0x77f656f60beca0eb,
                0x494aa01bdf32468d,
            ],
        });

        let mut none_count = 0;

        for _ in 0..100 {
            let square_root = square.sqrt();
            if square_root.is_none().into() {
                none_count += 1;
            } else {
                assert_eq!(square_root.unwrap() * square_root.unwrap(), square);
            }
            square -= Scalar::ONE;
        }

        assert_eq!(49, none_count);

        for _ in 0..1000 {
            // Ensure sqrt(a^2) = a or -a
            let a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let a_new = a.square().sqrt().unwrap();
            assert!(a_new == a || a_new == -a);
        }

        for _ in 0..1000 {
            // Ensure sqrt(a)^2 = a for random a
            let a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let sqrt = a.sqrt();
            if sqrt.is_some().into() {
                assert_eq!(sqrt.unwrap().square(), a);
            }
        }
    }

    #[test]
    fn double() {
        let a = Scalar(blst_fr {
            l: [
                0x1fff3231233ffffd,
                0x4884b7fa00034802,
                0x998c4fefecbc4ff3,
                0x1824b159acc50562,
            ],
        });

        assert_eq!(a.double(), a + a);

        for _ in 0..1000 {
            // Ensure doubling a is equivalent to adding a to itself.
            let a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let mut b = a;
            b.add_assign(&a);
            assert_eq!(a.double(), b);
        }
    }

    #[test]
    fn ordering() {
        assert_eq!(Scalar::ZERO, Scalar::ZERO,);
        assert!(
            Scalar(blst_fr {
                l: [9999, 9997, 9998, 9999],
            }) < Scalar(blst_fr {
                l: [9999, 9997, 9999, 9999],
            })
        );
        assert!(
            Scalar(blst_fr {
                l: [9, 9999, 9999, 9997],
            }) < Scalar(blst_fr {
                l: [9999, 9999, 9999, 9997],
            })
        );
    }

    #[test]
    fn from_u64() {
        let a = Scalar::from(100);
        let mut expected_bytes = [0u8; 32];
        expected_bytes[0] = 100;
        assert_eq!(a.to_repr(), expected_bytes);
    }

    #[test]
    fn is_odd() {
        assert!(bool::from(Scalar::from(0).is_even()));
        assert!(bool::from(Scalar::from(1).is_odd()));
        assert!(bool::from(Scalar::from(324834872).is_even()));
        assert!(bool::from(Scalar::from(324834873).is_odd()));
    }

    #[test]
    fn is_zero() {
        assert!(bool::from(Scalar::from(0).is_zero()));
        assert!(!bool::from(Scalar::from(1).is_zero()));
        assert!(!bool::from(Scalar(blst_fr { l: [0, 0, 1, 0] }).is_zero()));
    }

    #[test]
    fn square() {
        for _ in 0..10000 {
            // Ensure that (a * a) = a^2
            let a = Scalar::try_from_rng(&mut SysRng).unwrap();

            let tmp = a.square();

            let mut tmp2 = a;
            tmp2.mul_assign(&a);

            assert_eq!(tmp, tmp2);
        }
    }

    #[test]
    fn pow() {
        for i in 0..1000 {
            // Exponentiate by various small numbers and ensure it consists with repeated
            // multiplication.
            let a = Scalar::try_from_rng(&mut SysRng).unwrap();
            let target = a.pow_vartime([i]);
            let mut c = Scalar::ONE;
            for _ in 0..i {
                c.mul_assign(&a);
            }
            assert_eq!(c, target);
        }

        for _ in 0..1000 {
            // Exponentiating by the modulus should have no effect in a prime field.
            let a = Scalar::try_from_rng(&mut SysRng).unwrap();

            assert_eq!(a, a.pow_vartime(MODULUS));
        }
    }

    #[test]
    fn root_of_unity() {
        assert_eq!(Scalar::S, 32);
        assert_eq!(Scalar::MULTIPLICATIVE_GENERATOR, Scalar::from(7));
        assert_eq!(
            Scalar::MULTIPLICATIVE_GENERATOR.pow_vartime([
                0xfffe5bfeffffffff,
                0x9a1d80553bda402,
                0x299d7d483339d808,
                0x73eda753
            ]),
            Scalar::ROOT_OF_UNITY
        );
        assert_eq!(
            Scalar::ROOT_OF_UNITY.pow_vartime([1 << Scalar::S]),
            Scalar::ONE
        );
    }

    #[test]
    fn repr_conversion() {
        let a = Scalar::from(1);
        let mut expected_bytes = [0u8; 32];
        expected_bytes[0] = 1;
        assert_eq!(a, Scalar::from_repr(a.to_repr()).unwrap());
        assert_eq!(a.to_repr(), expected_bytes);
        assert_eq!(a, Scalar::from_repr(expected_bytes).unwrap());

        let a = Scalar::from(12);
        let mut expected_bytes = [0u8; 32];
        expected_bytes[0] = 12;
        assert_eq!(a, Scalar::from_repr(a.to_repr()).unwrap());
        assert_eq!(a.to_repr(), expected_bytes);
        assert_eq!(a, Scalar::from_repr(expected_bytes).unwrap());
    }

    #[test]
    fn m1_inv_bug() {
        // This fails on aarch64-darwin.
        let bad = Scalar::ZERO - Scalar::from(7);

        let inv = bad.invert().unwrap();
        let check = inv * bad;
        assert_eq!(Scalar::ONE, check);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn m1_inv_bug_more() {
        let mut bad = alloc::vec::Vec::new();
        for i in 1..1000000 {
            // Ensure that a * a^-1 = 1
            let a = Scalar::ZERO - Scalar::from(i);
            let ainv = a.invert().unwrap();
            let check = a * ainv;
            let one = Scalar::ONE;

            if check != one {
                bad.push((i, a));
            }
        }
        assert_eq!(0, bad.len());
    }

    // fn scalar_from_u64s(parts: [u64; 4]) -> Scalar {
    //     let mut le_bytes = [0u8; 32];
    //     le_bytes[0..8].copy_from_slice(&parts[0].to_le_bytes());
    //     le_bytes[8..16].copy_from_slice(&parts[1].to_le_bytes());
    //     le_bytes[16..24].copy_from_slice(&parts[2].to_le_bytes());
    //     le_bytes[24..32].copy_from_slice(&parts[3].to_le_bytes());
    //     let mut repr = <Scalar as PrimeField>::Repr::default();
    //     repr.as_mut().copy_from_slice(&le_bytes[..]);
    //     Scalar::from_repr_vartime(repr).expect("u64s exceed BLS12-381 scalar field modulus")
    // }

    #[test]
    #[cfg(feature = "alloc")]
    fn m1_inv_bug_special() {
        let maybe_bad = [Scalar(blst_fr {
            l: [
                0xb3fb72ea181b4e82,
                0x9435fcaf3a85c901,
                0x9eaf4fa6b9635037,
                0x2164d020b3bd14cc,
            ],
        })];

        let mut yep_bad = alloc::vec::Vec::new();

        for a in maybe_bad.iter() {
            let ainv = a.invert().unwrap();
            let check = ainv * a;
            let one = Scalar::ONE;

            if check != one {
                yep_bad.push(a);
            }
        }
        assert_eq!(0, yep_bad.len());
    }
}
