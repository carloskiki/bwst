pub mod g1;
pub mod g2;

pub mod bindings {
    #![allow(non_camel_case_types)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub struct MLResult(bindings::blst_fp12);
