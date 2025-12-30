use core::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use group::{Group, ff::PrimeField};
use rand_core::TryRngCore;
use subtle::{Choice, ConstantTimeEq};

use crate::{bindings, scalar::Scalar};

#[derive(Debug, Default, Clone, Copy)]
pub struct G2Projective(bindings::blst_p2);

impl Add for G2Projective {
    type Output = Self;

    #[allow(clippy::op_ref)]
    fn add(self, other: Self) -> Self::Output {
        self + &other
    }
}

impl Add<&G2Projective> for G2Projective {
    type Output = Self;

    fn add(mut self, other: &G2Projective) -> Self::Output {
        self += other;
        self
    }
}

impl AddAssign for G2Projective {
    fn add_assign(&mut self, other: Self) {
        *self += &other;
    }
}

impl AddAssign<&G2Projective> for G2Projective {
    fn add_assign(&mut self, other: &G2Projective) {
        // Safety: It is safe to call with `out` being one of the parameters.
        unsafe { bindings::blst_p2_add_or_double(&mut self.0, &self.0, &other.0) };
    }
}

impl Neg for G2Projective {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        // Safety: binding call with valid arguments.
        unsafe { bindings::blst_p2_cneg(&mut self.0, true) };
        self
    }
}

impl Neg for &G2Projective {
    type Output = G2Projective;

    fn neg(self) -> Self::Output {
        -*self
    }
}

impl Sub for G2Projective {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl Sub<&G2Projective> for G2Projective {
    type Output = Self;

    fn sub(self, other: &G2Projective) -> Self::Output {
        self + (-other)
    }
}

impl SubAssign for G2Projective {
    fn sub_assign(&mut self, other: Self) {
        *self += -other;
    }
}

impl SubAssign<&G2Projective> for G2Projective {
    fn sub_assign(&mut self, other: &G2Projective) {
        *self += -other;
    }
}

impl Sum for G2Projective {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, item| acc + item)
    }
}

impl<'a> Sum<&'a G2Projective> for G2Projective {
    fn sum<I: Iterator<Item = &'a G2Projective>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, item| acc + item)
    }
}

impl Mul<Scalar> for G2Projective {
    type Output = Self;

    fn mul(self, rhs: Scalar) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Scalar> for G2Projective {
    type Output = Self;

    fn mul(mut self, rhs: &Scalar) -> Self::Output {
        self *= rhs;
        self
    }
}

impl MulAssign<Scalar> for G2Projective {
    fn mul_assign(&mut self, rhs: Scalar) {
        *self *= &rhs;
    }
}

impl MulAssign<&Scalar> for G2Projective {
    fn mul_assign(&mut self, rhs: &Scalar) {
        // Safety: It is safe to call with `out` being one of the parameters.
        unsafe {
            bindings::blst_p2_mult(
                &mut self.0,
                &self.0,
                rhs.to_repr().as_ptr(),
                Scalar::NUM_BITS as usize,
            )
        };
    }
}

impl PartialEq for G2Projective {
    fn eq(&self, other: &Self) -> bool {
        // Safety: bindings call with valid arguments.
        unsafe { bindings::blst_p2_is_equal(&self.0, &other.0) }
    }
}
impl Eq for G2Projective {}
impl ConstantTimeEq for G2Projective {
    fn ct_eq(&self, other: &Self) -> Choice {
        ((self == other) as u8).into()
    }
}

impl Group for G2Projective {
    type Scalar = Scalar;

    fn try_from_rng<R: TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        let mut out = Self::default().0;
        let mut msg = [0u8; 64];
        rng.try_fill_bytes(&mut msg)?;
        const DST: [u8; 16] = [0; 16];
        const AUG: [u8; 16] = [0; 16];

        // Safety: bindings call with valid arguments.
        unsafe {
            bindings::blst_encode_to_g2(
                &mut out,
                msg.as_ptr(),
                msg.len(),
                DST.as_ptr(),
                DST.len(),
                AUG.as_ptr(),
                AUG.len(),
            )
        };

        Ok(G2Projective(out))
    }

    fn identity() -> Self {
        Default::default()
    }

    fn generator() -> Self {
        // Safety: bindings call returning a constant generator point.
        G2Projective(unsafe { *bindings::blst_p2_generator() })
    }

    fn is_identity(&self) -> Choice {
        // Safety: bindings call with valid argument.
        unsafe { Choice::from(bindings::blst_p2_is_inf(&self.0) as u8) }
    }

    fn double(&self) -> Self {
        let mut out = Self::default().0;
        // Safety: bindings call with valid arguments.
        unsafe { bindings::blst_p2_double(&mut out, &self.0) };
        G2Projective(out)
    }
}
