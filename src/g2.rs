use core::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use group::{Group, GroupEncoding, ff::PrimeField};
use rand_core::TryRngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::{
    bindings::{self, blst_p2},
    scalar::Scalar,
};

#[derive(Debug, Default, Clone, Copy)]
#[repr(transparent)]
pub struct G2Projective(blst_p2);

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

impl ConditionallySelectable for G2Projective {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let select_fp2 =
            |a: &bindings::blst_fp2, b: &bindings::blst_fp2, choice: Choice| -> bindings::blst_fp2 {
                let select_fp = |a: &bindings::blst_fp, b: &bindings::blst_fp, choice: Choice| -> bindings::blst_fp {
                    bindings::blst_fp {
                        l: [
                            u64::conditional_select(&a.l[0], &b.l[0], choice),
                            u64::conditional_select(&a.l[1], &b.l[1], choice),
                            u64::conditional_select(&a.l[2], &b.l[2], choice),
                            u64::conditional_select(&a.l[3], &b.l[3], choice),
                            u64::conditional_select(&a.l[4], &b.l[4], choice),
                            u64::conditional_select(&a.l[5], &b.l[5], choice),
                        ],
                    }
                };
                bindings::blst_fp2 {
                    fp: [select_fp(&a.fp[0], &b.fp[0], choice), select_fp(&a.fp[1], &b.fp[1], choice)],
                }
            };

        G2Projective(blst_p2 {
            x: select_fp2(&a.0.x, &b.0.x, choice),
            y: select_fp2(&a.0.y, &b.0.y, choice),
            z: select_fp2(&a.0.z, &b.0.z, choice),
        })
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Compressed(pub [u8; 96]);

impl Default for Compressed {
    fn default() -> Self {
        Compressed([0u8; 96])
    }
}

impl AsRef<[u8]> for Compressed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for Compressed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl GroupEncoding for G2Projective {
    type Repr = Compressed;

    fn from_bytes(bytes: &Self::Repr) -> subtle::CtOption<Self> {
        Self::from_bytes_unchecked(bytes).and_then(|point| {
            // Safety: bindings call with valid arguments.
            CtOption::new(
                point,
                (unsafe { bindings::blst_p2_in_g2(&point.0) } as u8).into(),
            )
        })
    }

    fn from_bytes_unchecked(bytes: &Self::Repr) -> subtle::CtOption<Self> {
        let mut affine = bindings::blst_p2_affine::default();
        // Safety: bindings call with valid arguments.
        let success = unsafe {
            bindings::blst_p2_uncompress(&mut affine, bytes.0.as_ptr())
                == bindings::BLST_ERROR::BLST_SUCCESS
        };
        let mut out = G2Projective::default();
        // Safety: bindings call with valid arguments.
        unsafe {
            bindings::blst_p2_from_affine(&mut out.0, &affine);
        }

        CtOption::new(out, Choice::from(success as u8))
    }

    fn to_bytes(&self) -> Self::Repr {
        let mut out = Compressed::default();
        // Safety: bindings call with valid arguments.
        unsafe {
            bindings::blst_p2_compress(out.0.as_mut_ptr(), &self.0);
        }
        out
    }
}

impl G2Projective {
    #[cfg(feature = "alloc")]
    pub fn linear_combination(points: &[Self], scalars: &[Scalar]) -> Self {
        use alloc::{vec, vec::Vec};

        let mut out = G2Projective::default().0;
        let len = points.len().min(scalars.len());
        let points = [points.as_ptr() as *const blst_p2, core::ptr::null()];
        let mut affines = Vec::with_capacity(len);
        // Safety: bindings call with valid arguments.
        // We do not need to set the length of `affines` because it is only used as a raw pointer,
        // and the points don't have a `Drop` implementation.
        unsafe { bindings::blst_p2s_to_affine(affines.as_mut_ptr(), &points[0], len) };
        let affines = [affines.as_ptr(), core::ptr::null()];

        let scalars = scalars
            .iter()
            .take(len)
            .flat_map(|s| s.to_repr())
            .collect::<Vec<_>>();
        let scalars = [scalars.as_ptr(), core::ptr::null()];

        // Safety: bindings call with valid arguments.
        let scratch_size = unsafe { bindings::blst_p2s_mult_pippenger_scratch_sizeof(len) };
        let mut scratch = vec![
            bindings::limb_t::default();
            scratch_size / core::mem::size_of::<bindings::limb_t>()
        ];

        // Safety: bindings call with valid arguments.
        unsafe {
            bindings::blst_p2s_mult_pippenger(
                &mut out,
                &affines[0],
                len,
                &scalars[0],
                Scalar::NUM_BITS as usize, // 255 fits in usize whatever the target ptr size
                scratch.as_mut_ptr(),
            );
        };

        G2Projective(out)
    }

    
    /// Hash to curve algorithm.
    // TODO: `hash2curve` traits when the crate does not depend on `elliptic-curve` anymore.
    pub fn hash_to_curve(msg: &[u8], dst: &[u8], aug: &[u8]) -> Self {
        let mut res = Self::identity();
        unsafe {
            bindings::blst_hash_to_g2(
                &mut res.0,
                msg.as_ptr(),
                msg.len(),
                dst.as_ptr(),
                dst.len(),
                aug.as_ptr(),
                aug.len(),
            );
        }
        res
    }
}
