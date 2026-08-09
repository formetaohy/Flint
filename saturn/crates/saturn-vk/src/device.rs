use std::sync::Arc;

use ash::vk;

use saturn_core::error::{Error, Result};

pub fn check(result: ash::prelude::VkResult<()>) -> Result<()> {
    result.map_err(|e| Error::Vulkan(e.to_string()))
}

pub fn check_value<T>(result: ash::prelude::VkResult<T>) -> Result<T> {
    result.map_err(|e| Error::Vulkan(e.to_string()))
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    pub validation: bool,
}

pub struct VkDevice {
    pub(crate) inner: Arc<VkDeviceInner>,
}

pub struct VkDeviceInner {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug_utils: Option<ash::ext::debug_utils::Instance>,
    pub messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub device: ash::Device,
    pub name: String,
    pub coop_shapes: Vec<saturn_core::CoopShape>,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub pool: vk::CommandPool,
    pub host_memory: u32,
    pub device_memory: u32,
    pub offset_alignment: u64,
    pub timestamp_period: f64,
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_invocations: u32,
    pub max_push_constants_size: u32,
}

impl VkDevice {
    pub fn open(options: &Options) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| Error::Vulkan(e.to_string()))?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"land")
            .application_version(0)
            .engine_name(c"land")
            .engine_version(0)
            .api_version(vk::API_VERSION_1_0);

        let layer_names = if options.validation {
            let layers = unsafe { entry.enumerate_instance_layer_properties() }
                .map_err(|e| Error::Vulkan(e.to_string()))?;
            let has_validation = layers.iter().any(|layer| {
                layer
                    .layer_name_as_c_str()
                    .is_ok_and(|name| name == c"VK_LAYER_KHRONOS_validation")
            });
            if has_validation {
                vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
            } else {
                log::warn!("validation requested but VK_LAYER_KHRONOS_validation is absent");
                vec![]
            }
        } else {
            vec![]
        };

        let extension_names = if options.validation {
            let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|e| Error::Vulkan(e.to_string()))?;
            let has_debug_utils = extensions.iter().any(|extension| {
                extension
                    .extension_name_as_c_str()
                    .is_ok_and(|name| name == c"VK_EXT_debug_utils")
            });
            if has_debug_utils {
                vec![c"VK_EXT_debug_utils".as_ptr()]
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| Error::Vulkan(e.to_string()))?;

        let (debug_utils, messenger) = if options.validation && !extension_names.is_empty() {
            let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vk_debug_callback));
            let messenger = unsafe { debug_utils.create_debug_utils_messenger(&info, None) }
                .map_err(|e| Error::Vulkan(e.to_string()))?;
            (Some(debug_utils), Some(messenger))
        } else {
            (None, None)
        };

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| Error::Vulkan(e.to_string()))?;
        let mut best: Option<(u32, u32, vk::PhysicalDevice)> = None;
        for &physical in &physical_devices {
            let props = unsafe { instance.get_physical_device_properties(physical) };
            let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
            if let Some((family, _)) = families
                .iter()
                .enumerate()
                .find(|(_, f)| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                let score = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                    _ => 0,
                };
                if best.is_none_or(|(best_score, _, _)| score > best_score) {
                    best = Some((score, family as u32, physical));
                }
            }
        }
        let (_, queue_family, physical) =
            best.ok_or(Error::NoBackend("vulkan"))?;
        let name = unsafe { instance.get_physical_device_properties(physical) }
            .device_name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8 as char)
            .collect::<String>();

        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let queue_ref = [queue_info];
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical)
                .map_err(|e| Error::Vulkan(e.to_string()))?
        };
        let has = |name: &str| {
            available_extensions
                .iter()
                .any(|ext| {
                    let bytes: Vec<u8> = ext
                        .extension_name
                        .iter()
                        .take_while(|&&c| c != 0)
                        .map(|&c| c as u8)
                        .collect();
                    bytes == name.as_bytes()
                })
        };
        let coop_matrix = has("VK_KHR_cooperative_matrix");
        let vulkan_memory_model = has("VK_KHR_vulkan_memory_model");
        let mut enabled = Vec::new();
        if coop_matrix {
            enabled.push("VK_KHR_cooperative_matrix");
        }
        if vulkan_memory_model {
            enabled.push("VK_KHR_vulkan_memory_model");
        }
        let enabled_refs: Vec<std::ffi::CString> = enabled
            .iter()
            .map(|name| std::ffi::CString::new(*name).expect("extension name"))
            .collect();
        let enabled_ptrs: Vec<*const i8> = enabled_refs
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_ref)
            .enabled_extension_names(&enabled_ptrs);
        let device = unsafe { instance.create_device(physical, &device_info, None) }
            .map_err(|e| Error::Vulkan(e.to_string()))?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        let host_memory = memory_props
            .memory_types
            .iter()
            .position(|t| {
                let f = t.property_flags;
                f.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                ) && !f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .or_else(|| {
                memory_props.memory_types.iter().position(|t| {
                    let f = t.property_flags;
                    f.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                })
            })
            .ok_or(Error::Vulkan(
                "no coherent host-visible memory type".to_string(),
            ))? as u32;
        let device_memory = find_memory_type(
            &memory_props.memory_types,
            vk::MemoryPropertyFlags::empty(),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(Error::Vulkan("no device-local memory type".to_string()))?;

        let limits = unsafe { instance.get_physical_device_properties(physical) }.limits;

        let offset_alignment = limits.min_storage_buffer_offset_alignment;
        let timestamp_period = limits.timestamp_period as f64;
        let max_workgroup_size = limits.max_compute_work_group_size;
        let max_workgroup_invocations = limits.max_compute_work_group_invocations;
        let max_push_constants_size = limits.max_push_constants_size;

        let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
        let coop_shapes = unsafe { cm.get_physical_device_cooperative_matrix_properties(physical) }
            .map(|props| {
                log::info!("vulkan: {} cooperative matrix shapes", props.len());
                props
                    .iter()
                    .filter_map(|p| {
                        Some(saturn_core::CoopShape {
                            a: component(p.a_type)?,
                            b: component(p.b_type)?,
                            c: component(p.c_type)?,
                            m: p.m_size,
                            n: p.n_size,
                            k: p.k_size,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(queue_family);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| Error::Vulkan(e.to_string()))?;

        log::info!("vulkan: opened {name} on queue family {queue_family}");
        let inner = Arc::new(VkDeviceInner {
            entry,
            instance,
            debug_utils,
            messenger,
            device,
            name,
            queue,
            queue_family,
            pool,
            host_memory,
            device_memory,
            offset_alignment,
            timestamp_period,
            max_workgroup_size,
            max_workgroup_invocations,
            max_push_constants_size,
            coop_shapes,
        });
        Ok(Self { inner })
    }
}

impl saturn_core::Device for VkDevice {
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn offset_alignment(&self) -> u64 {
        self.inner.offset_alignment
    }

    fn create_buffer(&self, spec: &saturn_core::BufferSpec) -> Result<Box<dyn saturn_core::Buffer>> {
        crate::buffer::VkBuffer::create(self, spec)
    }

    fn create_kernel(&self, spec: &saturn_core::KernelSpec) -> Result<Box<dyn saturn_core::Kernel>> {
        crate::kernel::VkKernel::create(self, spec)
    }

    fn encoder(&self) -> Result<Box<dyn saturn_core::CommandEncoder>> {
        let cmd = unsafe {
            self.inner.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.inner.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| Error::Vulkan(e.to_string()))?[0];
        check(unsafe {
            self.inner
                .device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
        })?;
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(
                crate::encoder::MAX_SETS_PER_POOL * crate::encoder::MAX_BINDINGS_PER_SET,
            );
        let pool = unsafe {
            self.inner.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(crate::encoder::MAX_SETS_PER_POOL)
                    .pool_sizes(&[pool_size]),
                None,
            )
        }
        .map_err(|e| Error::Vulkan(e.to_string()))?;
        Ok(Box::new(crate::encoder::VkEncoder {
            inner: self.inner.clone(),
            cmd,
            active: true,
            pools: vec![pool],
            current_pool: pool,
            sets_allocated: 0,
            bound: None,
        }))
    }

    fn coop_supported(&self, shape: saturn_core::CoopShape) -> bool {
        self.inner.coop_shapes.contains(&shape)
    }

    fn create_timestamp_set(&self, capacity: u32) -> Result<Box<dyn saturn_core::TimestampSet>> {
        crate::query::VkTimestampSet::create(self, capacity)
    }

    fn timestamp_period_ns(&self) -> f64 {
        self.inner.timestamp_period
    }

    fn submit(&self, encoder: Box<dyn saturn_core::CommandEncoder>) -> Result<Box<dyn saturn_core::Submission>> {
        let any: Box<dyn std::any::Any> = encoder;
        let mut encoder = any.downcast::<crate::encoder::VkEncoder>().map_err(|_| {
            Error::EncoderTypeMismatch {
                expected: std::any::type_name::<crate::encoder::VkEncoder>(),
                actual: "unknown",
            }
        })?;
        let inner = self.inner.clone();
        check(unsafe { inner.device.end_command_buffer(encoder.cmd) })?;
        let fence = unsafe { inner.device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| Error::Vulkan(e.to_string()))?;
        let cmd_ref = [encoder.cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_ref);
        check(unsafe { inner.device.queue_submit(inner.queue, &[submit_info], fence) })?;
        let cmd = encoder.take_cmd();
        let pools = encoder.take_pools();
        Ok(Box::new(crate::encoder::VkSubmission {
            inner,
            cmd,
            fence,
            pools,
            waited: std::cell::Cell::new(false),
        }))
    }
}

impl Drop for VkDeviceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.pool, None);
            if let (Some(debug_utils), Some(messenger)) = (&self.debug_utils, self.messenger) {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn component(ty: vk::ComponentTypeKHR) -> Option<saturn_core::Scalar> {
    use saturn_core::Scalar;
    match ty {
        vk::ComponentTypeKHR::FLOAT16 => Some(Scalar::F16),
        vk::ComponentTypeKHR::FLOAT32 => Some(Scalar::F32),
        vk::ComponentTypeKHR::UINT8 => Some(Scalar::U8),
        vk::ComponentTypeKHR::SINT8 => Some(Scalar::I8),
        _ => None,
    }
}

fn find_memory_type(
    types: &[vk::MemoryType],
    wanted: vk::MemoryPropertyFlags,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let flags = wanted | required;
    types
        .iter()
        .position(|t| t.property_flags & flags == flags)
        .map(|i| i as u32)
}

unsafe extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe {
        if data.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr((*data).p_message)
                .to_string_lossy()
                .into_owned()
        }
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("vulkan: {message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("vulkan: {message}");
    } else {
        log::debug!("vulkan: {message}");
    }
    vk::FALSE
}
