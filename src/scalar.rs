use core::ops::{Add, AddAssign, Neg};

use group::ff::{Field, PrimeField};

use crate::bindings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scalar(bindings::blst_fr);

impl Add for Scalar {
    type Output = Self;

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

impl Field for Scalar {
    const ZERO: Self;

    const ONE: Self;

    fn try_from_rng<R: rand_core::TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        todo!()
    }

    fn square(&self) -> Self {
        todo!()
    }

    fn double(&self) -> Self {
        todo!()
    }

    fn invert(&self) -> subtle::CtOption<Self> {
        todo!()
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (subtle::Choice, Self) {
        todo!()
    }
}

impl PrimeField for Scalar {
    type Repr;

    fn from_repr(repr: Self::Repr) -> subtle::CtOption<Self> {
        todo!()
    }

    fn to_repr(&self) -> Self::Repr {
        todo!()
    }

    fn is_odd(&self) -> subtle::Choice {
        todo!()
    }

    const MODULUS: &'static str;

    const NUM_BITS: u32;

    const CAPACITY: u32;

    const TWO_INV: Self;

    const MULTIPLICATIVE_GENERATOR: Self;

    const S: u32;

    const ROOT_OF_UNITY: Self;

    const ROOT_OF_UNITY_INV: Self;

    const DELTA: Self;
}
