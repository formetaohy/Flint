use ash::vk;

fn main() {
    let entry = unsafe { ash::Entry::load().expect("entry") };
    let instance = unsafe {
        entry
            .create_instance(&vk::InstanceCreateInfo::default(), None)
            .expect("instance")
    };
    let phys = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("enumerate")
            .into_iter()
            .find(|p| {
                instance.get_physical_device_properties(*p).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .or_else(|| {
                instance.enumerate_physical_devices().expect("enum").first().copied()
            })
            .expect("no device")
    };
    let props0 = unsafe { instance.get_physical_device_properties(phys) };
    let name = props0.device_name;
    println!("device: {:?}", unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) });
    let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
    let props = unsafe { cm.get_physical_device_cooperative_matrix_properties(phys) }
        .expect("query");
    println!("coop matrix combos: {}", props.len());
    for p in &props {
        println!(
            "  scope={:?} M={} N={} K={} A={:?} B={:?} C={:?} Result={:?} saturating={}",
            p.scope,
            p.m_size,
            p.n_size,
            p.k_size,
            p.a_type,
            p.b_type,
            p.c_type,
            p.result_type,
            p.saturating_accumulation == vk::TRUE
        );
    }
    unsafe { instance.destroy_instance(None) };
}
