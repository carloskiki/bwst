#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod g1;
pub mod g2;
pub mod miller_loop;
pub mod scalar;

pub use group;
// FIXME: This should not be re-exported
pub use subtle;

pub(crate) mod bindings {
    #![allow(non_camel_case_types)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
