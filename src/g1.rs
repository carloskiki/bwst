use crate::bindings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G1Affine(bindings::blst_p1_affine);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G1Projective(bindings::blst_p1);
