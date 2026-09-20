//! Minimal NVML device telemetry. GPU memory is device-wide, not process-local.
use anyhow::{Result, ensure};
use libloading::Library;
use serde::Serialize;
use std::ffi::c_void;
#[repr(C)]
#[derive(Default, Clone, Copy, Serialize)]
pub struct Memory {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}
#[repr(C)]
#[derive(Default, Clone, Copy, Serialize)]
pub struct Utilization {
    pub gpu: u32,
    pub memory: u32,
}
pub struct Nvml {
    _library: Library,
    handle: *mut c_void,
    memory: unsafe extern "C" fn(*mut c_void, *mut Memory) -> u32,
    util: unsafe extern "C" fn(*mut c_void, *mut Utilization) -> u32,
    shutdown: unsafe extern "C" fn() -> u32,
}
impl Nvml {
    pub fn new() -> Result<Self> {
        unsafe {
            let library = Library::new(if cfg!(windows) {
                "nvml.dll"
            } else {
                "libnvidia-ml.so.1"
            })?;
            let init: libloading::Symbol<unsafe extern "C" fn() -> u32> =
                library.get(b"nvmlInit_v2\0")?;
            ensure!(init() == 0, "NVML initialization failed");
            let get: libloading::Symbol<unsafe extern "C" fn(u32, *mut *mut c_void) -> u32> =
                library.get(b"nvmlDeviceGetHandleByIndex_v2\0")?;
            let mut handle = std::ptr::null_mut();
            ensure!(get(0, &mut handle) == 0, "NVML device unavailable");
            Ok(Self {
                handle,
                memory: *library.get(b"nvmlDeviceGetMemoryInfo\0")?,
                util: *library.get(b"nvmlDeviceGetUtilizationRates\0")?,
                shutdown: *library.get(b"nvmlShutdown\0")?,
                _library: library,
            })
        }
    }
    pub fn memory(&self) -> Result<Memory> {
        let mut m = Memory::default();
        unsafe {
            ensure!(
                (self.memory)(self.handle, &mut m) == 0,
                "NVML memory query failed"
            );
        }
        Ok(m)
    }
    pub fn utilization(&self) -> Result<Utilization> {
        let mut u = Utilization::default();
        unsafe {
            ensure!(
                (self.util)(self.handle, &mut u) == 0,
                "NVML utilization query failed"
            );
        }
        Ok(u)
    }
}
impl Drop for Nvml {
    fn drop(&mut self) {
        unsafe {
            (self.shutdown)();
        }
    }
}

#[cfg(windows)]
pub fn process_stats() -> Option<(u64, f64)> {
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        faults: u32,
        peak_working: usize,
        working: usize,
        peak_paged: usize,
        paged: usize,
        peak_nonpaged: usize,
        nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
        private: usize,
    }
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessTimes(
            p: *mut c_void,
            c: *mut FileTime,
            e: *mut FileTime,
            k: *mut FileTime,
            u: *mut FileTime,
        ) -> i32;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(p: *mut c_void, c: *mut Counters, n: u32) -> i32;
    }
    unsafe {
        let p = GetCurrentProcess();
        let mut m = Counters {
            cb: std::mem::size_of::<Counters>() as u32,
            ..Default::default()
        };
        let (mut c, mut e, mut k, mut u) = (
            FileTime::default(),
            FileTime::default(),
            FileTime::default(),
            FileTime::default(),
        );
        if GetProcessMemoryInfo(p, &mut m, std::mem::size_of::<Counters>() as u32) == 0
            || GetProcessTimes(p, &mut c, &mut e, &mut k, &mut u) == 0
        {
            return None;
        }
        let ticks = ((k.high as u64) << 32 | k.low as u64) + ((u.high as u64) << 32 | u.low as u64);
        Some((m.working as u64, ticks as f64 / 1e7))
    }
}
#[cfg(not(windows))]
pub fn process_stats() -> Option<(u64, f64)> {
    None
}
