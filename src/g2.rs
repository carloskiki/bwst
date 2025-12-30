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
pub struct Projective(pub(crate) blst_p2);

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
        unsafe { bindings::blst_p2_add_or_double(&mut self.0, &self.0, &other.0) };
    }
}

impl Neg for Projective {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        // Safety: binding call with valid arguments.
        unsafe { bindings::blst_p2_cneg(&mut self.0, true) };
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
            bindings::blst_p2_mult(
                &mut self.0,
                &self.0,
                rhs.to_repr().as_ptr(),
                Scalar::NUM_BITS as usize,
            )
        };
    }
}

impl PartialEq for Projective {
    fn eq(&self, other: &Self) -> bool {
        // Safety: bindings call with valid arguments.
        unsafe { bindings::blst_p2_is_equal(&self.0, &other.0) }
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
        let select_fp2 = |a: &bindings::blst_fp2,
                          b: &bindings::blst_fp2,
                          choice: Choice|
         -> bindings::blst_fp2 {
            let select_fp = |a: &bindings::blst_fp,
                             b: &bindings::blst_fp,
                             choice: Choice|
             -> bindings::blst_fp {
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
                fp: [
                    select_fp(&a.fp[0], &b.fp[0], choice),
                    select_fp(&a.fp[1], &b.fp[1], choice),
                ],
            }
        };

        Projective(blst_p2 {
            x: select_fp2(&a.0.x, &b.0.x, choice),
            y: select_fp2(&a.0.y, &b.0.y, choice),
            z: select_fp2(&a.0.z, &b.0.z, choice),
        })
    }
}

impl Group for Projective {
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

        Ok(Projective(out))
    }

    fn identity() -> Self {
        Default::default()
    }

    fn generator() -> Self {
        // Safety: bindings call returning a constant generator point.
        Projective(unsafe { *bindings::blst_p2_generator() })
    }

    fn is_identity(&self) -> Choice {
        // Safety: bindings call with valid argument.
        unsafe { Choice::from(bindings::blst_p2_is_inf(&self.0) as u8) }
    }

    fn double(&self) -> Self {
        let mut out = Self::default().0;
        // Safety: bindings call with valid arguments.
        unsafe { bindings::blst_p2_double(&mut out, &self.0) };
        Projective(out)
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

impl GroupEncoding for Projective {
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
        let mut out = Projective::default();
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

impl Projective {
    #[cfg(feature = "alloc")]
    pub fn linear_combination(points: &[Self], scalars: &[Scalar]) -> Self {
        use alloc::vec::Vec;

        let len = points.len().min(scalars.len());
        if len == 0 {
            return Projective::identity();
        }
        
        let mut out = Projective::default().0;
        let points = [points.as_ptr() as *const blst_p2, core::ptr::null()];
        let mut affines = Vec::with_capacity(len);
        // Safety: bindings call with valid arguments.
        // We do not need to set the length of `affines` because it is only used as a raw pointer,
        // and the points don't have a `Drop` implementation.
        unsafe { bindings::blst_p2s_to_affine(affines.as_mut_ptr(), points.as_ptr(), len) };
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
            bindings::blst_p2s_mult_pippenger(
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

#[cfg(test)]
mod tests {
    use getrandom::SysRng;
    use group::{Group, GroupEncoding};

    use super::Projective;
    use crate::{
        bindings::{blst_fp, blst_fp2, blst_fr},
        scalar::Scalar,
    };

    fn is_on_curve(point: &Projective) -> bool {
        // Safety: bindings call with valid arguments.
        unsafe { crate::bindings::blst_p2_on_curve(&point.0) }
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

        let generator = Projective::generator();
        assert!(is_on_curve(&generator));

        // Reject point on isomorphic twist (b = 3 * (u + 1))
        {
            let p_affine = crate::bindings::blst_p2_affine {
                x: blst_fp2 {
                    fp: [
                        blst_fp {
                            l: [
                                0xa757072d9fa35ba9,
                                0xae3fb2fb418f6e8a,
                                0xc1598ec46faa0c7c,
                                0x7a17a004747e3dbe,
                                0xcc65406a7c2e5a73,
                                0x10b8c03d64db4d0c,
                            ],
                        },
                        blst_fp {
                            l: [
                                0xd30e70fe2f029778,
                                0xda30772df0f5212e,
                                0x5b47a9ff9a233a50,
                                0xfb777e5b9b568608,
                                0x789bac1fec71a2b9,
                                0x1342f02e2da54405,
                            ],
                        },
                    ],
                },
                y: blst_fp2 {
                    fp: [
                        blst_fp {
                            l: [
                                0xfe0812043de54dca,
                                0xe455171a3d47a646,
                                0xa493f36bc20be98a,
                                0x663015d9410eb608,
                                0x78e82a79d829a544,
                                0x40a00545bb3c1e,
                            ],
                        },
                        blst_fp {
                            l: [
                                0x4709802348e79377,
                                0xb5ac4dc9204bcfbd,
                                0xda361c97d02f42b2,
                                0x15008b1dc399e8df,
                                0x68128fd0548a3829,
                                0x16a613db5c873aaa,
                            ],
                        },
                    ],
                },
            };
            let p2 = {
                let mut out = Projective::default();
                // Safety: bindings call with valid arguments.
                unsafe { crate::bindings::blst_p2_from_affine(&mut out.0, &p_affine) };
                out
            };
            let compressed = p2.to_bytes();

            assert!(bool::from(Projective::from_bytes(&compressed).is_none()));
            assert!(bool::from(
                Projective::from_bytes_unchecked(&compressed).is_some()
            ));
        }

        // Reject point on a twist (b = 2 * (u + 1))
        {
            let p_affine = crate::bindings::blst_p2_affine {
                x: blst_fp2 {
                    fp: [
                        blst_fp {
                            l: [
                                0xf4fdfe95a705f917,
                                0xc2914df688233238,
                                0x37c6b12cca35a34b,
                                0x41abba710d6c692c,
                                0xffcc4b2b62ce8484,
                                0x6993ec01b8934ed,
                            ],
                        },
                        blst_fp {
                            l: [
                                0xb94e92d5f874e26,
                                0x44516408bc115d95,
                                0xe93946b290caa591,
                                0xa5a0c2b7131f3555,
                                0x83800965822367e7,
                                0x10cf1d3ad8d90bfa,
                            ],
                        },
                    ],
                },
                y: blst_fp2 {
                    fp: [
                        blst_fp {
                            l: [
                                0xbf00334c79701d97,
                                0x4fe714f9ff204f9a,
                                0xab70b28002f3d825,
                                0x5a9171720e73eb51,
                                0x38eb4fd8d658adb7,
                                0xb649051bbc1164d,
                            ],
                        },
                        blst_fp {
                            l: [
                                0x9225814253d7df75,
                                0xc196c2513477f887,
                                0xe05e2fbd15a804e0,
                                0x55f2b8efad953e04,
                                0x7379345eda55265e,
                                0x377f2e6208fd4cb,
                            ],
                        },
                    ],
                },
            };
            let p2 = {
                let mut out = Projective::default();
                // Safety: bindings call with valid arguments.
                unsafe { crate::bindings::blst_p2_from_affine(&mut out.0, &p_affine) };
                out
            };
            let compressed = p2.to_bytes();

            assert!(bool::from(Projective::from_bytes(&compressed).is_none()));
            assert!(bool::from(
                Projective::from_bytes_unchecked(&compressed).is_some()
            ));
        }
    }

    #[test]
    fn equality() {
        let a = Projective::generator();
        let b = Projective::identity();

        assert_eq!(a, a);
        assert_eq!(b, b);
        assert_ne!(a, b);
        assert_ne!(b, a);
    }

    #[test]
    fn doubling() {
        let tmp = Projective::identity().double();
        assert!(bool::from(tmp.is_identity()));
        assert!(is_on_curve(&tmp));
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
        let length_mismatch =
            Projective::linear_combination(&points, &scalars);
        assert_eq!(length_mismatch, Projective::identity());

        let points = vec![Projective::generator(); 2];
        let length_mismatch =
            Projective::linear_combination(&points, &scalars);
        assert_eq!(length_mismatch, Projective::generator());
    }
}
