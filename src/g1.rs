use core::{
    iter::Sum,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

use group::Group;
use rand_core::TryRngCore;
use subtle::Choice;

use crate::{bindings, scalar::Scalar};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G1Affine(bindings::blst_p1_affine);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G1Projective(bindings::blst_p1);

// macro_rules! arithmetic_impl {
//     ($group:ty, $trait:ident<$rhs:ty>, $method:ident, $binding:ident) => {
//         arithmetic_impl!(@impl $group, $trait<$rhs>, $method, $binding);
//         arithmetic_impl!(@impl $group, $trait<&$rhs>, $method, $binding);
//     };
//
//     (@impl $group:ty, $trait:ident<$rhs:ty>, $method:ident, $binding:ident) => {
//         impl $trait<$rhs> for $group {
//             type Output = Self;
//
//             fn $method(mut self, other: $rhs) -> Self::Output {
//                 // Safety: It is safe to call with `out` being one of the parameters.
//                 unsafe { bindings::$binding(&mut self.0, &self.0, &other.0) };
//                 self
//             }
//         }
//     }
// }
//
// arithmetic_impl!(G1Projective, Add<Self>, add, blst_p1_add_or_double);
// arithmetic_impl!(G1Projective, Sub<Self>, sub, blst_p1_);

impl Add for G1Projective {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self + &other
    }
}

impl Add<&G1Projective> for G1Projective {
    type Output = Self;

    fn add(mut self, other: &G1Projective) -> Self::Output {
        self += other;
        self
    }
}

impl AddAssign for G1Projective {
    fn add_assign(&mut self, other: Self) {
        *self += &other;
    }
}

impl AddAssign<&G1Projective> for G1Projective {
    fn add_assign(&mut self, other: &G1Projective) {
        // Safety: It is safe to call with `out` being one of the parameters.
        unsafe { bindings::blst_p1_add_or_double(&mut self.0, &self.0, &other.0) };
    }
}

impl Neg for G1Projective {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        // Safety: binding call with valid arguments.
        unsafe { bindings::blst_p1_cneg(&mut self.0, true) };
        self
    }
}

impl Neg for &G1Projective {
    type Output = G1Projective;

    fn neg(self) -> Self::Output {
        -*self
    }
}

impl Sub for G1Projective {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl Sub<&G1Projective> for G1Projective {
    type Output = Self;

    fn sub(self, other: &G1Projective) -> Self::Output {
        self + (-other)
    }
}

impl SubAssign for G1Projective {
    fn sub_assign(&mut self, other: Self) {
        *self += -other;
    }
}

impl SubAssign<&G1Projective> for G1Projective {
    fn sub_assign(&mut self, other: &G1Projective) {
        *self += -other;
    }
}

impl Sum for G1Projective {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, item| acc + item)
    }
}

impl<'a> Sum<&'a G1Projective> for G1Projective {
    fn sum<I: Iterator<Item = &'a G1Projective>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, item| acc + item)
    }
}

impl Group for G1Projective {
    type Scalar = Scalar;

    fn try_from_rng<R: TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        todo!()
    }

    fn identity() -> Self {
        todo!()
    }

    fn generator() -> Self {
        todo!()
    }

    fn is_identity(&self) -> Choice {
        todo!()
    }

    fn double(&self) -> Self {
        todo!()
    }
}
