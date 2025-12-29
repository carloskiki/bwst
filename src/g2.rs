use crate::bindings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G2Affine(bindings::blst_p2_affine);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G2Projective(bindings::blst_p2);
