use core::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use group::ff::{Field, PrimeField};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::bindings::{self, blst_fr};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Scalar(blst_fr);

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

    fn try_from_rng<R: rand_core::TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        loop {
            let mut scalar = bindings::blst_scalar { b: [0u8; 32] };
            rng.try_fill_bytes(&mut scalar.b)?;

            // Mask away the unused most-significant bits.
            scalar.b[31] &= 0xff >> (256 - Scalar::NUM_BITS as usize);

            if unsafe { bindings::blst_scalar_fr_check(&scalar) } {
                let out_ptr: *mut blst_fr =
                    &mut scalar as *mut bindings::blst_scalar as *mut blst_fr;
                // Safety: both types have the same size, and are `repr(C)`. The out pointer can be
                // the same as the input pointer.
                unsafe { bindings::blst_fr_from_scalar(out_ptr, &scalar) };
                // Safety: aligned, valid, and unique pointer.
                return Ok(Scalar(unsafe { *out_ptr }));
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
        let out_ptr: *mut blst_fr = &mut scalar as *mut bindings::blst_scalar as *mut blst_fr;
        // Safety: both types have the same size, and are `repr(C)`. The out pointer can be
        // the same as the input pointer.
        unsafe { bindings::blst_fr_from_scalar(out_ptr, &scalar) };
        // Safety: aligned, valid, and unique pointer.
        let out = Scalar(unsafe { *out_ptr });
        CtOption::new(
            out,
            // Safety: scalar is valid, and respects the ffi contract.
            (unsafe { bindings::blst_scalar_fr_check(&scalar) } as u8).into(),
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
