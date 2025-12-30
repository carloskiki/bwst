#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod g1;
pub mod g2;
pub mod miller_loop;
pub mod scalar;

pub mod bindings {
    #![allow(non_camel_case_types)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
