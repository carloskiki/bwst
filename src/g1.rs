use group::Group;

use crate::bindings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G1Affine(bindings::blst_p1_affine);

impl Group for G1Affine {
    type Scalar;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G1Projective(bindings::blst_p1);
