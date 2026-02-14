use core::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use group::{Group, GroupEncoding, ff::PrimeField};
use rand_core::TryRng;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::{
    bindings::{self, blst_p1},
    scalar::Scalar,
};

#[derive(Debug, Default, Clone, Copy)]
#[repr(transparent)]
pub struct Projective(pub(crate) blst_p1);

impl Add for Projective {
    type Output = Self;

    #[allow(clippy::op_ref)]
    fn add(self, other: Self) -> Self::Output {
        self + &other
    }
}

impl Add<&Projective> for Projective {
    type Output = Self;

    fn add(mut self, other: &Projective) -> Self::Output {
        self += other;
        self
    }
}

impl AddAssign for Projective {
    fn add_assign(&mut self, other: Self) {
        *self += &other;
    }
}

impl AddAssign<&Projective> for Projective {
    fn add_assign(&mut self, other: &Projective) {
        // Safety: It is safe to call with `out` being one of the parameters.
        unsafe { bindings::blst_p1_add_or_double(&mut self.0, &self.0, &other.0) };
    }
}

impl Neg for Projective {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        // Safety: binding call with valid arguments.
        unsafe { bindings::blst_p1_cneg(&mut self.0, true) };
        self
    }
}

impl Neg for &Projective {
    type Output = Projective;

    fn neg(self) -> Self::Output {
        -*self
    }
}

impl Sub for Projective {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl Sub<&Projective> for Projective {
    type Output = Self;

    fn sub(self, other: &Projective) -> Self::Output {
        self + (-other)
    }
}

impl SubAssign for Projective {
    fn sub_assign(&mut self, other: Self) {
        *self += -other;
    }
}

impl SubAssign<&Projective> for Projective {
    fn sub_assign(&mut self, other: &Projective) {
        *self += -other;
    }
}

impl Sum for Projective {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, item| acc + item)
    }
}

impl<'a> Sum<&'a Projective> for Projective {
    fn sum<I: Iterator<Item = &'a Projective>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, item| acc + item)
    }
}

impl Mul<Scalar> for Projective {
    type Output = Self;

    fn mul(self, rhs: Scalar) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Scalar> for Projective {
    type Output = Self;

    fn mul(mut self, rhs: &Scalar) -> Self::Output {
        self *= rhs;
        self
    }
}

impl MulAssign<Scalar> for Projective {
    fn mul_assign(&mut self, rhs: Scalar) {
        *self *= &rhs;
    }
}

impl MulAssign<&Scalar> for Projective {
    fn mul_assign(&mut self, rhs: &Scalar) {
        // Safety: It is safe to call with `out` being one of the parameters.
        unsafe {
            bindings::blst_p1_mult(
                &mut self.0,
                &self.0,
                rhs.to_repr().as_ptr(),
                Scalar::NUM_BITS as usize, // 255 fits in usize whatever the target ptr size
            )
        };
    }
}

impl PartialEq for Projective {
    fn eq(&self, other: &Self) -> bool {
        // Safety: bindings call with valid arguments.
        unsafe { bindings::blst_p1_is_equal(&self.0, &other.0) }
    }
}
impl Eq for Projective {}
impl ConstantTimeEq for Projective {
    fn ct_eq(&self, other: &Self) -> Choice {
        ((self == other) as u8).into()
    }
}

impl ConditionallySelectable for Projective {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let select_fp =
            |a: &bindings::blst_fp, b: &bindings::blst_fp, choice: Choice| -> bindings::blst_fp {
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

        Projective(blst_p1 {
            x: select_fp(&a.0.x, &b.0.x, choice),
            y: select_fp(&a.0.y, &b.0.y, choice),
            z: select_fp(&a.0.z, &b.0.z, choice),
        })
    }
}

impl Group for Projective {
    type Scalar = Scalar;

    fn try_from_rng<R: TryRng + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        let mut out = Self::default().0;
        let mut msg = [0u8; 64];
        rng.try_fill_bytes(&mut msg)?;
        const DST: [u8; 16] = [0; 16];
        const AUG: [u8; 16] = [0; 16];

        // Safety: bindings call with valid arguments.
        unsafe {
            bindings::blst_encode_to_g1(
                &mut out,
                msg.as_ptr(),
                msg.len(),
                DST.as_ptr(),
                DST.len(),
                AUG.as_ptr(),
                AUG.len(),
            )
        };

        Ok(Projective(out))
    }

    fn identity() -> Self {
        Default::default()
    }

    fn generator() -> Self {
        // Safety: bindings call returning a constant generator point.
        Projective(unsafe { *bindings::blst_p1_generator() })
    }

    fn is_identity(&self) -> Choice {
        // Safety: bindings call with valid argument.
        unsafe { Choice::from(bindings::blst_p1_is_inf(&self.0) as u8) }
    }

    fn double(&self) -> Self {
        let mut out = Self::default().0;
        // Safety: bindings call with valid arguments.
        unsafe { bindings::blst_p1_double(&mut out, &self.0) };
        Projective(out)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Compressed(pub [u8; 48]);

impl Default for Compressed {
    fn default() -> Self {
        Compressed([0u8; 48])
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

impl GroupEncoding for Projective {
    type Repr = Compressed;

    fn from_bytes(bytes: &Self::Repr) -> subtle::CtOption<Self> {
        Self::from_bytes_unchecked(bytes).and_then(|point| {
            // Safety: bindings call with valid arguments.
            CtOption::new(
                point,
                (unsafe { bindings::blst_p1_in_g1(&point.0) } as u8).into(),
            )
        })
    }

    fn from_bytes_unchecked(bytes: &Self::Repr) -> subtle::CtOption<Self> {
        let mut affine = bindings::blst_p1_affine::default();
        // Safety: bindings call with valid arguments.
        let success = unsafe {
            bindings::blst_p1_uncompress(&mut affine, bytes.0.as_ptr())
                == bindings::BLST_ERROR::BLST_SUCCESS
        };
        let mut out = Projective::default();
        // Safety: bindings call with valid arguments.
        unsafe {
            bindings::blst_p1_from_affine(&mut out.0, &affine);
        }

        CtOption::new(out, Choice::from(success as u8))
    }

    fn to_bytes(&self) -> Self::Repr {
        let mut out = Compressed::default();
        // Safety: bindings call with valid arguments.
        unsafe {
            bindings::blst_p1_compress(out.0.as_mut_ptr(), &self.0);
        }
        out
    }
}

impl Projective {
    pub const IDENTITY: Self = Self(blst_p1 {
        x: bindings::blst_fp { l: [0; 6] },
        y: bindings::blst_fp { l: [0; 6] },
        z: bindings::blst_fp { l: [0; 6] },
    });
    
    #[cfg(feature = "alloc")]
    pub fn linear_combination(points: &[Self], scalars: &[Scalar]) -> Self {
        use alloc::vec::Vec;

        let len = points.len().min(scalars.len());
        if len == 0 {
            return Projective::identity();
        }
        
        let mut out = Projective::default().0;
        let points = [points.as_ptr() as *const blst_p1, core::ptr::null()];
        let mut affines = Vec::with_capacity(len);
        // Safety: bindings call with valid arguments.
        // We do not need to set the length of `affines` because it is only used as a raw pointer,
        // and the points don't have a `Drop` implementation.
        unsafe { bindings::blst_p1s_to_affine(affines.as_mut_ptr(), points.as_ptr(), len) };
        let affines = [affines.as_ptr(), core::ptr::null()];

        let scalars = scalars
            .iter()
            .take(len)
            .flat_map(|s| s.to_repr())
            .collect::<Vec<_>>();
        let scalars = [scalars.as_ptr(), core::ptr::null()];

        // Safety: bindings call with valid arguments.
        let scratch_size = unsafe { bindings::blst_p1s_mult_pippenger_scratch_sizeof(len) };
        let mut scratch: Vec<bindings::limb_t> =
            Vec::with_capacity(scratch_size / core::mem::size_of::<bindings::limb_t>());

        // Safety: bindings call with valid arguments.
        unsafe {
            bindings::blst_p1s_mult_pippenger(
                &mut out,
                &affines[0],
                len,
                &scalars[0],
                Scalar::NUM_BITS as usize, // 255 fits in usize whatever the target ptr size
                scratch.as_mut_ptr(), // We do not set the length of `scratch` because its elements
                                      // don't have a `Drop` implementation.
            );
        };

        Projective(out)
    }

    /// Hash to curve algorithm.
    // TODO: `hash2curve` traits when the crate does not depend on `elliptic-curve` anymore.
    pub fn hash_to_curve(msg: &[u8], dst: &[u8], aug: &[u8]) -> Self {
        let mut res = Self::identity();
        unsafe {
            bindings::blst_hash_to_g1(
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

#[cfg(test)]
mod tests {
    use getrandom::SysRng;
    use group::{Group, GroupEncoding};

    use super::Projective;
    use crate::{
        bindings::{blst_fp, blst_fr},
        scalar::Scalar,
    };

    fn is_on_curve(point: &Projective) -> bool {
        // Safety: bindings call with valid arguments.
        unsafe { crate::bindings::blst_p1_on_curve(&point.0) }
    }

    #[test]
    fn identity() {
        {
            let z = Projective::identity();
            assert!(bool::from(z.is_identity()));
        }

        // Negation edge case with zero.
        {
            let mut z = Projective::identity();
            z = -z;
            assert!(bool::from(z.is_identity()));
        }

        // Doubling edge case with zero.
        {
            let mut z = Projective::identity();
            z = z.double();
            assert!(bool::from(z.is_identity()));
        }

        // Addition edge cases with zero
        {
            let mut r = Projective::try_from_rng(&mut SysRng).unwrap();
            let rcopy = r;
            r += &Projective::identity();
            assert_eq!(r, rcopy);

            let mut z = Projective::identity();
            z += Projective::identity();
            assert!(bool::from(z.is_identity()));

            let mut z2 = z;
            z2 += &r;

            assert_eq!(z2, r);
        }
    }

    #[test]
    fn on_curve() {
        assert!(is_on_curve(&Projective::identity()));

        let mut generator = Projective::generator();
        assert!(is_on_curve(&generator));

        let new_x = blst_fp {
            l: [
                0xba7afa1f9a6fe250,
                0xfa0f5b595eafe731,
                0x3bdc477694c306e7,
                0x2149be4b3949fa24,
                0x64aa6e0649b2078c,
                0x12b108ac33643c3e,
            ],
        };

        generator.0.x = new_x;
        assert!(!is_on_curve(&generator));
    }

    #[test]
    fn equality() {
        let a = Projective::generator();
        let b = Projective::identity();

        assert_eq!(a, a);
        assert_eq!(b, b);
        assert_ne!(a, b);
        assert_ne!(b, a);

        let mut c = Projective::generator() - Projective::identity();

        assert_eq!(a, c);
        assert_eq!(c, a);
        assert_ne!(b, c);
        assert_ne!(c, b);

        c.0.y = (-c).0.y;
        assert!(is_on_curve(&c));

        assert_ne!(a, c);
        assert_ne!(b, c);
        assert_ne!(c, a);
        assert_ne!(c, b);
    }

    #[test]
    fn doubling() {
        {
            let a = Projective::generator().double().double(); // 4P
            let b = Projective::generator().double(); // 2P
            let c = a + b; // 4P + 2P = 6P
            let d = Projective::generator() * Scalar::from(6u64); // 6P
            assert_eq!(c, d);
        }
    }

    #[test]
    fn addition() {
        {
            let a = Projective::identity();
            let b = Projective::identity();
            let c = a + b;
            assert!(bool::from(c.is_identity()));
        }
        {
            let a = Projective::identity();
            let b = Projective::generator();
            let c1 = a + b;
            let c2 = b + a;
            assert_eq!(c2, Projective::generator());
            assert_eq!(c1, c2);
        }
        {
            let a = Projective::generator().double().double(); // 4P
            let b = Projective::generator().double(); // 2P
            let c = a + b;

            let mut d = Projective::generator();
            for _ in 0..5 {
                d += Projective::generator();
            }
            assert_eq!(c, d);
        }
    }

    #[test]
    fn subtraction() {
        let a = Projective::generator().double();
        assert_eq!(a + (-a), Projective::identity());
        assert_eq!(a + (-a), a - a);
    }

    #[test]
    fn multiplication() {
        let g = Projective::generator();
        let a = Scalar(blst_fr {
            l: [
                0x2b568297a56da71c,
                0xd8c39ecb0ef375d1,
                0x435c38da67bfbf96,
                0x8088a05026b659b2,
            ],
        });
        let b = Scalar(blst_fr {
            l: [
                0x785fdd9b26ef8b85,
                0xc997f25837695c18,
                0x4c8dbc39e7b756c1,
                0x70d9b6cc6d87df20,
            ],
        });
        let c = a * b;

        assert_eq!((g * a) * b, g * c);
    }

    #[test]
    fn serialization() {
        for _ in 0..100 {
            let el = Projective::try_from_rng(&mut SysRng).unwrap();
            let c = el.to_bytes();
            assert_eq!(Projective::from_bytes(&c).unwrap(), el);
            assert_eq!(Projective::from_bytes_unchecked(&c).unwrap(), el);
        }
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn linear_combination() {
        use alloc::{vec, vec::Vec};
        use group::ff::Field;

        const SIZE: usize = 10;
        let points: Vec<Projective> = (0..SIZE)
            .map(|_| Projective::try_from_rng(&mut SysRng))
            .collect::<Result<_, _>>()
            .unwrap();
        let scalars: Vec<Scalar> = (0..SIZE)
            .map(|_| Scalar::try_from_rng(&mut SysRng))
            .collect::<Result<_, _>>()
            .unwrap();

        let mut naive = points[0] * scalars[0];
        for i in 1..SIZE {
            naive += points[i] * scalars[i];
        }

        let pippenger = Projective::linear_combination(points.as_slice(), scalars.as_slice());
        assert_eq!(naive, pippenger);

        let points: Vec<Projective> = Vec::new();
        let scalars: Vec<Scalar> = Vec::new();
        let empty_linear_combination =
            Projective::linear_combination(points.as_slice(), scalars.as_slice());
        assert_eq!(empty_linear_combination, Projective::identity());

        let scalars = vec![Scalar::ONE; 1];
        let length_mismatch = Projective::linear_combination(&points, &scalars);
        assert_eq!(length_mismatch, Projective::identity());

        let points = vec![Projective::generator(); 2];
        let length_mismatch = Projective::linear_combination(&points, &scalars);
        assert_eq!(length_mismatch, Projective::generator());
    }
}
