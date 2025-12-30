use core::ops::{Mul, MulAssign};

use crate::{
    bindings::{self, blst_fp12},
    g1, g2,
};

#[derive(Clone, Copy, Debug)]
pub struct Result(blst_fp12);

impl Eq for Result {}

impl PartialEq for Result {
    fn eq(&self, other: &Self) -> bool {
        unsafe { bindings::blst_fp12_is_equal(&self.0, &other.0) }
    }
}

impl Mul for Result {
    type Output = Self;

    #[allow(clippy::op_ref)]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Result> for Result {
    type Output = Self;

    fn mul(mut self, rhs: &Result) -> Self::Output {
        self *= rhs;
        self
    }
}

impl MulAssign for Result {
    fn mul_assign(&mut self, rhs: Result) {
        *self *= &rhs;
    }
}

impl MulAssign<&Result> for Result {
    fn mul_assign(&mut self, rhs: &Result) {
        // Safety: binding call with valid arguments.
        unsafe { bindings::blst_fp12_mul(&mut self.0, &self.0, &rhs.0) };
    }
}

impl Result {
    pub fn miller_loop(p: &g1::Projective, q: &g2::Projective) -> Result {
        let p_affine = {
            let mut out = bindings::blst_p1_affine::default();
            // Safety: binding call with valid arguments.
            unsafe { bindings::blst_p1_to_affine(&mut out, &p.0) }
            out
        };
        let q_affine = {
            let mut out = bindings::blst_p2_affine::default();
            // Safety: binding call with valid arguments.
            unsafe { bindings::blst_p2_to_affine(&mut out, &q.0) }
            out
        };

        // Safety: binding call with valid arguments.
        let mut out = unsafe { *bindings::blst_fp12_one() };
        // Safety: binding call with valid arguments.
        unsafe { bindings::blst_miller_loop(&mut out, &q_affine, &p_affine) }
        Result(out)
    }

    pub fn final_verify(&self, other: &Result) -> bool {
        // Safety: binding call with valid arguments.
        unsafe { bindings::blst_fp12_finalverify(&self.0, &other.0) }
    }
}
