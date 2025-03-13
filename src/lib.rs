//! Radix sort algorithm used for sorting keys of type `u32`.

pub mod get_subgroup_size;
pub use get_subgroup_size::*;

use std::ops::Range;

use bevy::{
    asset::{RenderAssetUsages, load_internal_asset},
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BufferUsages,
            CachedComputePipelineId, CachedPipelineState, CommandEncoder, ComputePass,
            ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, PushConstantRange,
            ShaderDefVal, ShaderStages,
            binding_types::{storage_buffer, storage_buffer_read_only},
        },
        renderer::RenderDevice,
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
    },
};

pub const NUMBER_OF_BYTES_PER_KEY: u32 = std::mem::size_of::<u32>() as u32;
/// The number of bits per pass that can be processed.
///
/// Due to the nature of the radix sort algorithm, the radix range should not be too large,
/// as the first step involves generating a histogram of radix counts.
/// Since `BYTE` is the smallest effective unit in a computer, 8 is a suitable number of bits for the radix (1 BYTE = 8 BIT).
///
/// Assuming the keys to be processed are of type `u32`, when selecting 8 bits as the radix size, 4 passes are required.
pub const NUMBER_OF_RADIX_BITS: u32 = 8;
/// The range of the radix, the range of the radix with 8 bits is [0, 255].
pub const NUMBER_OF_RADIX: u32 = 1 << NUMBER_OF_RADIX_BITS;

/// `WARP` is a term used by Nvidia to refer to a group of parallel threads that execute the same instruction set within a time slice.
/// `WARP` also has synonymous terms such as `WAVEFRONT` (AMD), `SIMD Group` (Apple), etc.
/// However, here it is collectively referred to as `Subgroup`.
///
/// The value of `subgroup_size` varies depending on the GPU manufacturer and model:
/// - Nvidia: The value of `subgroup_size` is 32 for all GPU models.
/// - AMD: For older GPU models, `subgroup_size` is 64, while for newer models, it can be either 32 or 64.
/// - Apple: Apple has not provided specific specifications, but based on testing, `subgroup_size` is 32 (Apple M1 Pro).
/// - Intel: There is no concept similar to `WARP`, and the size of thread groups is dynamic.
///     Please refer to the documentation for specific details.
///
/// The value of [`NUMBER_OF_THREADS_PER_WORKGROUP`] should be a multiple of `subgroup_size`.
///
/// For most hardware devices, the value of `subgroup_size` is 32.
///
/// 256 is a multiple of 32, and the range of the `u8` type is also 256,
/// so choosing 256 as the value of [`NUMBER_OF_THREADS_PER_WORKGROUP`] is appropriate.
pub const NUMBER_OF_THREADS_PER_WORKGROUP: u32 = NUMBER_OF_RADIX;
/// The row size of the keys processed by each workgroup.
///
/// The memory layout of the keys processed by a workgroup is as follows:
///
/// ```text
///              thread 0       thread 1                       thread 255                                                 
///         ┌───────────────┬───────────────┬───────────────┬───────────────┐                                             
///  row 0  │      K_0      │      K_1      │     ...       │     K_255     │                                             
///         ├───────────────┼───────────────┼───────────────┼───────────────┤                                             
///  row 1  │     V_256     │     V_257     │     ...       │     V_511     │                                             
///         ├───────────────┴───────────────┴───────────────┴───────────────┤                                             
///   ...   │                              ...                              │                                             
///         ├───────────────┬───────────────┬───────────────┬───────────────┤                                             
/// row 15  │     V_256n    │    V_256n+1   │     ...       │  V_256n+255   │                                             
///         └───────────────┴───────────────┴───────────────┴───────────────┘                                             
/// ```
///
/// ## Why 3/7/15 row?
///
/// The number is good for avoiding `Bank Conflict` in the `shared memory` of the GPU.
///
/// TODO: Refactor to automatically select configurations to adapt to different hardware devices for maximum performance.
pub const NUMBER_OF_ROWS_PER_WORKGROUP: u32 = 7;

pub const RADIX_SORT_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(174050053373014597864115292867874370814);

/// When pass is even:
///
/// ```wgsl
/// @binding(0) var<storage, read_write> global_keys_i: array<u32>;
/// ```
///
/// When pass is odd:
///
/// ```wgsl
/// @binding(3) var<storage, read_write> global_keys_o: array<u32>;
/// ```
pub const EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE: Handle<ShaderStorageBuffer> =
    Handle::weak_from_u128(271984633984723648237498237498274982749);
/// When pass is even:
///
/// ```wgsl
/// @binding(1) var<storage, read_write> global_vals_i: array<u32>;
/// ```
///
/// When pass is odd:
///
/// ```wgsl
/// @binding(4) var<storage, read_write> global_vals_o: array<u32>;
/// ```
pub const EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE: Handle<ShaderStorageBuffer> =
    Handle::weak_from_u128(198374692837469283746928374692837469283);
/// ```wgsl
/// @binding(2) var<storage, read_write> global_blocks: array<u32>;
/// ```
pub const GLOBAL_BLOCKS_STORAGE_BUFFER_HANDLE: Handle<ShaderStorageBuffer> =
    Handle::weak_from_u128(287346928374692837469283746928374692837);
/// When pass is even:
///
/// ```wgsl
/// @binding(3) var<storage, read_write> global_keys_o: array<u32>;
/// ```
///
/// When pass is odd:
///
/// ```wgsl
/// @binding(0) var<storage, read_write> global_keys_i: array<u32>;
/// ```
pub const ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE: Handle<ShaderStorageBuffer> =
    Handle::weak_from_u128(340282366910938463463374607431768211234);
/// When pass is even:
///
/// ```wgsl
/// @binding(4) var<storage, read_write> global_vals_o: array<u32>;
/// ```
///
/// When pass is odd:
///
/// ```wgsl
/// @binding(1) var<storage, read_write> global_vals_i: array<u32>;
/// ```
pub const ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE: Handle<ShaderStorageBuffer> =
    Handle::weak_from_u128(123456789012345678901234567890123456789);

pub struct RadixSortPlugin {
    pub settings: RadixSortSettings,
}

impl Plugin for RadixSortPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            RADIX_SORT_SHADER_HANDLE,
            "radix_sort.wgsl",
            Shader::from_wgsl
        );

        create_shader_storage_buffers(app, self.settings.max_number_of_keys());

        app.insert_resource(self.settings);
        app.sub_app_mut(RenderApp)
            .insert_resource(self.settings)
            .add_systems(
                Render,
                RadixSortBindGroup::initialize
                    .in_set(RenderSet::PrepareBindGroups)
                    .run_if(not(resource_exists::<RadixSortBindGroup>)),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<RadixSortPipeline>();
    }
}

fn create_shader_storage_buffers(app: &mut App, max_number_of_keys: u32) {
    let number_of_keys_per_scatter_block =
        NUMBER_OF_THREADS_PER_WORKGROUP * NUMBER_OF_ROWS_PER_WORKGROUP;
    let max_number_of_blks = max_number_of_keys.div_ceil(number_of_keys_per_scatter_block);

    let mut sbufs = app
        .world_mut()
        .resource_mut::<Assets<ShaderStorageBuffer>>();

    let usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let size = (max_number_of_keys * NUMBER_OF_BYTES_PER_KEY) as usize;

    let mut eve_global_keys_buf =
        ShaderStorageBuffer::with_size(size, RenderAssetUsages::default());
    eve_global_keys_buf.buffer_description.label =
        Some("radix_sort: global_keys buffer - input when even-pass, output when odd-pass");
    eve_global_keys_buf.buffer_description.usage = usages;

    let mut eve_global_vals_buf =
        ShaderStorageBuffer::with_size(size, RenderAssetUsages::default());
    eve_global_vals_buf.buffer_description.label =
        Some("radix_sort: global_vals buffer - input when even-pass, output when odd-pass");
    eve_global_vals_buf.buffer_description.usage = usages;
    eve_global_vals_buf.buffer_description.mapped_at_creation = true;

    let mut global_blocks_buf = ShaderStorageBuffer::with_size(
        (max_number_of_blks * NUMBER_OF_RADIX * NUMBER_OF_BYTES_PER_KEY) as usize,
        RenderAssetUsages::default(),
    );
    global_blocks_buf.buffer_description.label = Some("radix_sort: global_blocks buffer");
    global_blocks_buf.buffer_description.usage = usages;

    let mut odd_global_keys_buf =
        ShaderStorageBuffer::with_size(size, RenderAssetUsages::default());
    odd_global_keys_buf.buffer_description.label =
        Some("radix_sort: global_keys buffer - input when odd-pass, output when even-pass");
    odd_global_keys_buf.buffer_description.usage = usages;

    let mut odd_global_vals_buf =
        ShaderStorageBuffer::with_size(size, RenderAssetUsages::default());
    odd_global_vals_buf.buffer_description.label =
        Some("radix_sort: global_vals buffer - input when odd-pass, output when even-pass");
    odd_global_vals_buf.buffer_description.usage = usages;
    odd_global_vals_buf.buffer_description.mapped_at_creation = true;

    sbufs.insert(
        EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id(),
        eve_global_keys_buf,
    );
    sbufs.insert(
        EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id(),
        eve_global_vals_buf,
    );
    sbufs.insert(GLOBAL_BLOCKS_STORAGE_BUFFER_HANDLE.id(), global_blocks_buf);
    sbufs.insert(
        ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id(),
        odd_global_keys_buf,
    );
    sbufs.insert(
        ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id(),
        odd_global_vals_buf,
    );
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct RadixSortSettings {
    max_number_of_keys: u32,
}

impl RadixSortSettings {
    pub fn max_number_of_keys(&self) -> u32 {
        self.max_number_of_keys
    }
}

impl From<u32> for RadixSortSettings {
    fn from(max_number_of_keys: u32) -> Self {
        Self { max_number_of_keys }
    }
}

/// ## Introduction
///
/// This implementation of the `radix-sort` algorithm is based on the paper:
/// [Fast 4-way parallel radix sorting on GPUs](http://www.sci.utah.edu/~csilva/papers/cgf.pdf)
///
/// The radix sort algorithm can be divided into 3 steps:
///
/// 1. Count the number of each radix in the `block` and generate a histogram (count_radix_pipeline).
/// 2. Perform prefix sum operations on the histogram (scan_upsweep_pipeline, scan_dnsweep_pipeline, scan_last_block_pipeline).
/// 3. Based on the histogram information, write the key values in the `block` to new ordered positions (scatter_pipeline).
///
/// ## Modification
///
/// I have made modifications to the __Step 2: Compute prefix sum on the histogram__ in the paper to improve performance.
///
/// The memory layout of `global_histogram_buffer` in the paper is as follows:
///
/// ```text
///               workgroup0      workgroup1        ...           workgroupN                                                                    
///           +---------------+---------------+---------------+---------------+                                                                 
/// radix 0   |      V_0      |      V_1      |     ...       |     V_n-1     |                                                                 
///           +---------------+---------------+---------------+---------------+                                                                 
/// radix 1   |      V_n      |      V_n+1    |     ...       |     V_2n-1    |                                                                 
///           +---------------+---------------+---------------+---------------+                                                                 
///  ...      |                              ...                              |                                                                 
///           +---------------+---------------+---------------+---------------+                                                                 
/// radix 255 |     V_255n    |     V_255n+1  |     ...       |    V_256n-1   |                                                                 
///           +---------------+---------------+---------------+---------------+                                                                 
/// ```
///
/// This memory layout is beneficial for __Step 2__ prefix sum calculations, but not for __Step 1__ and __Step 3__,
/// because the memory access to `global_histogram_buffer` in __Step 1__ and __Step 3__ is `Non-Coalesced Memory Access`.
///
/// `Non-Coalesced Memory Access` can cause significant delays, leading to a drastic decrease in the performance of `radix_sort`.
///  To address this issue, I redesigned the memory layout of `global_histogram_buffer`:
///
/// ```text
///                  radix 0         radix 1          ...            radix 255                                                                    
///             +---------------+---------------+---------------+---------------+                                                                 
/// workgroup0  |      V_0      |      V_1      |     ...       |     V_255     |                                                                 
///             +---------------+---------------+---------------+---------------+                                                                 
/// workgroup1  |     V_256     |     V_257     |     ...       |     V_511     |                                                                 
///             +---------------+---------------+---------------+---------------+                                                                 
///    ...      |                              ...                              |                                                                 
///             +---------------+---------------+---------------+---------------+                                                                 
/// workgroupN  |     V_256n    |    V_256n+1   |     ...       |  V_256n+255   |                                                                 
///             +---------------+---------------+---------------+---------------+                                                         
/// ```
///
/// This solves the delay problem caused by `Non-Coalesced Memory Access`,
/// but brings a new problem: how to perform prefix sum calculations on such a memory layout?
///
/// The idea is simple: change the unit of the prefix sum operation from `radix` to `histogram`,
/// and perform the prefix sum operation on the `histogram`.
/// This means accumulating `histogram-workgroup0` to `histogram-workgroup1`, `histogram-workgroup1` to `histogram-workgroup2`, and so on.
///
/// Of course, we cannot use a serial prefix sum as it would degrade performance.
/// Therefore, we use the [Blelloch](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
/// parallel prefix sum algorithm, which consists of an up-sweep and a down-sweep step:
///
/// ### Up-Sweep
///
/// ```text
///           ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐                                                       
///           │H0│ │H1│ │H2│ │H3│ │H4│ │H5│ │H6│ │H7│                                                       
///           └┬─┘ └┬─┘ └┬─┘ └┬─┘ └┬─┘ └┬─┘ └┬─┘ └┬─┘                                                       
///            └───┬▼─┐  └───┬▼─┐  └───┬▼─┐  └───┬▼─┐                                                       
/// Round 0        │H1│      │H3│      │H5│      │H7│                                                       
///                └┴─┴──────┴┬─┘      └──┴──────┴┬─┘                                                       
///                          ┌▼─┐                ┌▼─┐                                                       
/// Round 1                  │H3│                │H7│                                                       
///                          └──┴────────────────┴┬─┘                                                       
///                                              ┌▼─┐                                                       
/// Round 2                                      │H7│                                                       
///                                              └──┘   
/// ```
///
/// ### Down-Sweep
///
/// ```text
///     Round 1   Round 0   Round 1                                                                          
///     ┌────┐    ┌────┐    ┌────┐                                                                           
/// ┌──┐│┌──┐│┌──┐│┌──┐│┌──┐│┌──┐│┌──┐ ┌──┐                                                                  
/// │H0│││H1│││H2│││H3│││H4│││H5│││H6│ │H7│                                                                  
/// └──┘│└──┘│└▲─┘│└──┘│└▲─┘│└──┘│└▲─┘ └──┘                                                                  
///     │┌──┐│ │  │┌──┐│ │  │┌──┐│ │   ┌──┐                                                                  
///     ││H1├┼─┘  ││H3├┼─┘  ││H5├┼─┘   │H7│                                                                  
///     │└──┘│    │└──┘│    │└▲─┘│     └──┘                                                                  
///     └────┘    │┌──┐│    └─┼──┘     ┌──┐                                                                  
///               ││H3├┼──────┘        │H7│                                                                  
///               │└──┘│               └──┘                                                                  
///               └────┘               ┌──┐                                                                  
///                                    │H7│                                                                  
///                                    └──┘  
/// ```
///
/// The performance of parallel algorithms on GPUs is often bandwidth-sensitive.
/// Therefore, the number of memory accesses is a crucial indicator of the algorithm's performance.
///
/// The memory access count of the `Blelloch` algorithm is quite high: 2R1W + 2R1W = 4R2W, which is a rather poor situation.
/// Moreover, since it executes on `device-memory`, it cannot hide latency through `shared memory`.
/// (General parallel prefix sum algorithms can achieve 2R1W memory access count, so the performance of `device-scope` `Blelloch`
/// is not good, only half of the pre-modified performance)
///
/// However, the performance degradation brought by `Blelloch` is not a big issue here,
/// because we can reduce the overall cost of __Step 2__ by reducing the number of `histograms`.
/// To reduce the number of `histograms`, the method is very straightforward:
/// increase the number of keys processed during the statistics phase (__Step 1__).
///
/// Previously, one workgroup processed 256 keys and output a `histogram` of length 256.
/// It can be modified to one workgroup processing 16*256 keys and outputting a `histogram` of length 256.
/// This reduces the number of `histograms` by 16x, thereby reducing the memory access overhead of __Step 2__.
///
/// ## Advanced?
///
/// In fact, the algorithm can be further optimized!
///
/// We can use the `decoupled look-back` algorithm to halve memory access overhead, achieving nearly 100% performance improvement.
/// However, the problem is that `decoupled look-back` relies on communication between `workgroups`,
/// and there are two ways to achieve `workgroup` communication:
///
/// 1. device-scope storage-buffer memory barrier.
/// 2. `forward progress guarantees`.
///
/// The rendering layer of `bevy` relies on `wgpu`, which is an implementation of the `webgpu` standard.
/// Unfortunately, `webgpu` does not support either of these methods.
/// This is mainly due to `Apple`'s `Metal` graphics API not supporting these two features,
/// which leads to `webgpu`, aiming for full compatibility, also not supporting them.
///
/// For more details, see [atomic concerns](https://github.com/gpuweb/gpuweb/issues/2229).
///
/// But fortunately, for the vast majority of desktop GPUs (Nvidia/AMD), `forward progress guarantees` are implicitly supported.
/// So if your target platform does not include `Apple` and mobile devices,
/// you can still use `webgpu` to implement the `decoupled look-back radix sort` algorithm.
///
/// However, for compatibility reasons, a more traditional radix sort algorithm is used here.
///
/// ## Tips
///
/// The number of pass these 3 steps are executed depends on the number of bits in the keys and the number of bits processed per pass.
/// For keys of type `u32` and `RADIX_BITS_PER_PASS` set to 8, 4 passes are required.
///
/// If the effective number of bits is less than 32 (but the key type is still `u32`), for example,
/// if all keys are less than 256, then only 1 pass is needed.
///
/// You can customize the number of pass and positions for processing the keys according to your specific requirements to improve performance.
#[derive(Resource, Debug, Clone)]
pub struct RadixSortPipeline {
    /// Generate a histogram by counting the number of each radix in the `block`
    /// The histogram's x-axis represents the radix, while the y-axis represents the number of each radix
    count_radix_pipeline: CachedComputePipelineId,
    /// Perform prefix sum (inclusive) operation on the histogram in a histogram-wise manner, divided into up-sweep and down-sweep steps.
    ///
    /// This is the up-sweep step.
    scan_upsweep_pipeline: CachedComputePipelineId,
    /// Perform prefix sum (inclusive) operation on the histogram in a histogram-wise manner, divided into up-sweep and down-sweep steps.
    ///
    /// This is the down-sweep step.
    scan_dnsweep_pipeline: CachedComputePipelineId,
    /// Perform prefix sum (exclusive) operation on the histogram of the last block.
    scan_last_block_pipeline: CachedComputePipelineId,
    /// Write the key values to new ordered positions based on the radix.
    scatter_pipeline: CachedComputePipelineId,
    /// The bindgroup layout is:
    ///
    /// ```wgsl
    /// @binding(0) var<storage, read_write> global_keys_i: array<u32>;
    /// @binding(1) var<storage, read_write> global_blocks: array<u32>;
    /// @binding(2) var<storage, read_write> global_keys_o: array<u32>;
    /// ```
    bind_group_layout: BindGroupLayout,
}

impl RadixSortPipeline {
    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}

impl FromWorld for RadixSortPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let subgroup_size = world.resource::<SubgroupSize>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "radix_sort bindgroup layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // Read unsorted(sub-sort) keys from this buffer
                    storage_buffer_read_only::<u32>(false),
                    // Read unsorted(sub-sort) vals from this buffer
                    storage_buffer_read_only::<u32>(false),
                    // Read/Write histograms of count of each radix
                    storage_buffer::<u32>(false),
                    // Write sorted(sub-sort) keys to this buffer
                    storage_buffer::<u32>(false),
                    // Write sorted(sub-sort) vals to this buffer
                    storage_buffer::<u32>(false),
                ),
            ),
        );

        let cdefs = vec![
            ShaderDefVal::UInt(
                "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
                NUMBER_OF_THREADS_PER_WORKGROUP,
            ),
            ShaderDefVal::UInt(
                "NUMBER_OF_ROWS_PER_WORKGROUP".into(),
                NUMBER_OF_ROWS_PER_WORKGROUP,
            ),
            ShaderDefVal::UInt("NUMBER_OF_RADIX".into(), NUMBER_OF_RADIX),
            ShaderDefVal::UInt("NUMBER_OF_RADIX_BITS".into(), NUMBER_OF_RADIX_BITS),
            ShaderDefVal::UInt(
                "NUMBER_OF_THREADS_PER_SUBGROUP".into(),
                subgroup_size.into(),
            ),
        ];

        let count_radix_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("radix_sort: count_radix pipeline".into()),
                layout: vec![bind_group_layout.clone()],
                push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
                shader: RADIX_SORT_SHADER_HANDLE,
                shader_defs: [cdefs.as_slice(), &["COUNT_RADIX_PIPELINE".into()]].concat(),
                entry_point: "main".into(),
                zero_initialize_workgroup_memory: false,
            });

        let scan_upsweep_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("radix_sort: scan_upsweep pipeline".into()),
                layout: vec![bind_group_layout.clone()],
                push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
                shader: RADIX_SORT_SHADER_HANDLE,
                shader_defs: [cdefs.as_slice(), &["SCAN_UP_SWEEP_PIPELINE".into()]].concat(),
                entry_point: "main".into(),
                zero_initialize_workgroup_memory: false,
            });

        let scan_dnsweep_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("radix_sort: scan_dnsweep pipeline".into()),
                layout: vec![bind_group_layout.clone()],
                push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
                shader: RADIX_SORT_SHADER_HANDLE,
                shader_defs: [cdefs.as_slice(), &["SCAN_DOWN_SWEEP_PIPELINE".into()]].concat(),
                entry_point: "main".into(),
                zero_initialize_workgroup_memory: false,
            });

        let scan_last_block_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("radix_sort: scan_last_block pipeline".into()),
                layout: vec![bind_group_layout.clone()],
                push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
                shader: RADIX_SORT_SHADER_HANDLE,
                shader_defs: [cdefs.as_slice(), &["SCAN_LAST_BLOCK_PIPELINE".into()]].concat(),
                entry_point: "main".into(),
                zero_initialize_workgroup_memory: false,
            });

        let scatter_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort: scatter pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: RADIX_SORT_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["SCATTER_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            count_radix_pipeline,
            scan_upsweep_pipeline,
            scan_dnsweep_pipeline,
            scan_last_block_pipeline,
            scatter_pipeline,
            bind_group_layout,
        }
    }
}

/// The radix sort algorithm requires multiple sub-sorts.
/// For example, for keys of type `u32` and [`NUMBER_OF_RADIX_BITS`] set to 8, 4 sub-sorts are needed.
///
/// Since the sub-sort algorithm is not an in-place sorting algorithm,
/// two [`Buffer`]s are needed to alternate as the input and output for key sorting.
/// Additionally, two [`BindGroup`]s are required to bind the two [`Buffer`]s to different input and output slots.
#[derive(Resource, Debug, Clone)]
pub struct RadixSortBindGroup {
    /// When pass is even, set this bind_group to compute pass
    eve_bind_group: BindGroup,
    /// When pass is odd, set this bind_group to compute pass
    odd_bind_group: BindGroup,
}

impl RadixSortBindGroup {
    pub fn initialize(
        mut commands: Commands,
        radix_sort_pipeline: Res<RadixSortPipeline>,
        radix_sort_settings: Res<RadixSortSettings>,
        render_device: Res<RenderDevice>,
        sbufs: Res<RenderAssets<GpuShaderStorageBuffer>>,
    ) {
        let bind_group_layout = radix_sort_pipeline.bind_group_layout();

        let eve_global_keys_buf = sbufs
            .get(EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let eve_global_vals_buf = sbufs
            .get(EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let global_blocks_buf = sbufs.get(GLOBAL_BLOCKS_STORAGE_BUFFER_HANDLE.id()).unwrap();
        let odd_global_keys_buf = sbufs
            .get(ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_global_vals_buf = sbufs
            .get(ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();

        // Initialize `eve_global_vals_buf`/`odd_global_vals_buf` with a sequence of natural numbers,
        // which is very useful as it can serve as the default index value for the first call.
        let init_vals: Vec<u32> = (0..radix_sort_settings.max_number_of_keys()).collect();
        let byte_size =
            (radix_sort_settings.max_number_of_keys() * NUMBER_OF_BYTES_PER_KEY) as usize;

        eve_global_vals_buf.buffer.slice(..).get_mapped_range_mut()[..byte_size]
            .copy_from_slice(bytemuck::cast_slice(&init_vals));
        eve_global_vals_buf.buffer.unmap();

        odd_global_vals_buf.buffer.slice(..).get_mapped_range_mut()[..byte_size]
            .copy_from_slice(bytemuck::cast_slice(&init_vals));
        odd_global_vals_buf.buffer.unmap();

        let eve_bind_group = render_device.create_bind_group(
            "radix_sort: bind_group for even-pass",
            bind_group_layout,
            &BindGroupEntries::sequential((
                eve_global_keys_buf.buffer.as_entire_binding(),
                eve_global_vals_buf.buffer.as_entire_binding(),
                global_blocks_buf.buffer.as_entire_binding(),
                odd_global_keys_buf.buffer.as_entire_binding(),
                odd_global_vals_buf.buffer.as_entire_binding(),
            )),
        );

        let odd_bind_group = render_device.create_bind_group(
            "radix_sort: bind_group for odd-pass",
            bind_group_layout,
            &BindGroupEntries::sequential((
                odd_global_keys_buf.buffer.as_entire_binding(),
                odd_global_vals_buf.buffer.as_entire_binding(),
                global_blocks_buf.buffer.as_entire_binding(),
                eve_global_keys_buf.buffer.as_entire_binding(),
                eve_global_vals_buf.buffer.as_entire_binding(),
            )),
        );

        let radix_sort_bind_group = Self {
            eve_bind_group,
            odd_bind_group,
        };

        commands.insert_resource(radix_sort_bind_group);
    }

    pub fn eve_bind_group(&self) -> &BindGroup {
        &self.eve_bind_group
    }

    pub fn odd_bind_group(&self) -> &BindGroup {
        &self.odd_bind_group
    }
}

const WORKGROUP_OFFSET_OFFSET: u32 = 0;
/// The number of keys to be sorted.
const NUMBER_OF_KEYS_OFFSET: u32 = 4;
/// The number of blocks(histogram) required.
///
/// `number_of_blks` = ceil(`number_of_keys` / [`NUMBER_OF_THREADS_PER_WORKGROUP`])
const NUMBER_OF_BLKS_OFFSET: u32 = 8;
/// The current `pass` index being processed. For `u32` type with 8-bit `radix`, it requires 4 passes to process.
/// So the valid range for `pass_index` is [0, 3].
///
/// Since we are using the LSD (Least Significant Digit) sorting method, the `pass_index` represents:
/// - `pass_index` = 0: Processing the least significant 8 bits of the `radix`,         0x000000XX
/// - `pass_index` = 1: Processing the second least significant 8 bits of the `radix`,  0x0000XX00
/// - `pass_index` = 2: Processing the second most significant 8 bits of the `radix`,   0x00XX0000
/// - `pass_index` = 3: Processing the most significant 8 bits of the `radix`,          0xXX000000
const PASS_INDEX_OFFSET: u32 = 12;
/// Used to control the step size of the prefix sum (inclusive) algorithm in step 2, up-sweep and down-sweep
const SWEEP_SIZE_OFFSET: u32 = 16;
/// Used to control whether to automatically write the index to `odd_global_vals_buf` in the 0th pass
const INIT_INDEX_OFFSET: u32 = 20;

const PUSH_CONSTANT_RANGES: PushConstantRange = PushConstantRange {
    stages: ShaderStages::COMPUTE,
    range: 0..24,
};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum LoadState {
    OnLoad,
    Loaded,
    Failed(String),
}

pub fn check_load_state(world: &World) -> LoadState {
    let pipeline_cache = world.resource::<PipelineCache>();
    let radix_sort_pipeline = world.resource::<RadixSortPipeline>();

    let (
        count_radix_pipeline_state,
        scan_upsweep_pipeline_state,
        scan_dnsweep_pipeline_state,
        scan_last_block_pipeline_state,
        scatter_pipeline_state,
    ) = (
        pipeline_cache.get_compute_pipeline_state(radix_sort_pipeline.count_radix_pipeline),
        pipeline_cache.get_compute_pipeline_state(radix_sort_pipeline.scan_upsweep_pipeline),
        pipeline_cache.get_compute_pipeline_state(radix_sort_pipeline.scan_dnsweep_pipeline),
        pipeline_cache.get_compute_pipeline_state(radix_sort_pipeline.scan_last_block_pipeline),
        pipeline_cache.get_compute_pipeline_state(radix_sort_pipeline.scatter_pipeline),
    );

    if let CachedPipelineState::Err(err) = count_radix_pipeline_state {
        return LoadState::Failed(format!("Failed to load count_radix_pipeline: {:?}", err));
    }

    if let CachedPipelineState::Err(err) = scan_upsweep_pipeline_state {
        return LoadState::Failed(format!("Failed to load scan_upsweep_pipeline: {:?}", err));
    }

    if let CachedPipelineState::Err(err) = scan_dnsweep_pipeline_state {
        return LoadState::Failed(format!("Failed to load scan_dnsweep_pipeline: {:?}", err));
    }

    if let CachedPipelineState::Err(err) = scan_last_block_pipeline_state {
        return LoadState::Failed(format!(
            "Failed to load scan_last_block_pipeline: {:?}",
            err
        ));
    }

    if let CachedPipelineState::Err(err) = scatter_pipeline_state {
        return LoadState::Failed(format!("Failed to load scatter_pipeline: {:?}", err));
    }

    if matches!(count_radix_pipeline_state, CachedPipelineState::Ok(_))
        && matches!(scan_upsweep_pipeline_state, CachedPipelineState::Ok(_))
        && matches!(scan_dnsweep_pipeline_state, CachedPipelineState::Ok(_))
        && matches!(scan_last_block_pipeline_state, CachedPipelineState::Ok(_))
        && matches!(scatter_pipeline_state, CachedPipelineState::Ok(_))
    {
        return LoadState::Loaded;
    }

    LoadState::OnLoad
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    encoder: &mut CommandEncoder,
    pipeline_cache: &PipelineCache,
    radix_sort_pipeline: &RadixSortPipeline,
    radix_bind_group: &RadixSortBindGroup,
    max_compute_workgroups_per_dimension: u32,
    number_of_keys: u32,
    pass_range: Range<u32>,
    init_index: bool,
    read_from_even: bool,
) {
    if number_of_keys < 2 {
        return;
    }

    let count_radix_pipeline = pipeline_cache
        .get_compute_pipeline(radix_sort_pipeline.count_radix_pipeline)
        .unwrap();
    let scan_upsweep_pipeline = pipeline_cache
        .get_compute_pipeline(radix_sort_pipeline.scan_upsweep_pipeline)
        .unwrap();
    let scan_dnsweep_pipeline = pipeline_cache
        .get_compute_pipeline(radix_sort_pipeline.scan_dnsweep_pipeline)
        .unwrap();
    let scan_last_block_pipeline = pipeline_cache
        .get_compute_pipeline(radix_sort_pipeline.scan_last_block_pipeline)
        .unwrap();
    let scatter_pipeline = pipeline_cache
        .get_compute_pipeline(radix_sort_pipeline.scatter_pipeline)
        .unwrap();

    let number_of_keys_per_scatter_block =
        NUMBER_OF_THREADS_PER_WORKGROUP * NUMBER_OF_ROWS_PER_WORKGROUP;
    let number_of_blks = number_of_keys.div_ceil(number_of_keys_per_scatter_block);

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("radix_sort compute pass"),
            ..default()
        });

        pass.set_pipeline(count_radix_pipeline);
        pass.set_push_constants(NUMBER_OF_KEYS_OFFSET, bytemuck::bytes_of(&number_of_keys));
        pass.set_push_constants(NUMBER_OF_BLKS_OFFSET, bytemuck::bytes_of(&number_of_blks));
        pass.set_push_constants(INIT_INDEX_OFFSET, bytemuck::bytes_of(&(init_index as u32)));

        for pass_index in pass_range {
            pass.set_push_constants(PASS_INDEX_OFFSET, bytemuck::bytes_of(&pass_index));

            // If read_from_even is true:
            //   pass_index == 0: `even_global_keys_buf`-> `odd_global_keys_buf`
            //   pass_index == 1: `odd_global_keys_buf` -> `even_global_keys_buf`
            //   pass_index == 2: `even_global_keys_buf`-> `odd_global_keys_buf`
            //   pass_index == 3: `odd_global_keys_buf` -> `even_global_keys_buf`
            // If read_from_even is false:
            //   pass_index == 0: `odd_global_keys_buf` -> `even_global_keys_buf`
            //   pass_index == 1: `even_global_keys_buf`-> `odd_global_keys_buf`
            //   pass_index == 2: `odd_global_keys_buf` -> `even_global_keys_buf`
            //   pass_index == 3: `even_global_keys_buf`-> `odd_global_keys_buf`
            if (pass_index + read_from_even as u32) % 2 == 0 {
                pass.set_bind_group(0, radix_bind_group.odd_bind_group(), &[]);
            } else {
                pass.set_bind_group(0, radix_bind_group.eve_bind_group(), &[]);
            }

            // 1. count radix histogram
            {
                pass.set_pipeline(count_radix_pipeline);

                dispatch_workgroup_ext(
                    &mut pass,
                    number_of_blks,
                    max_compute_workgroups_per_dimension,
                    WORKGROUP_OFFSET_OFFSET,
                );
            }

            // 2. scan blocks
            {
                // scan up sweep(inclusive)
                pass.set_pipeline(scan_upsweep_pipeline);
                let num_round = log2_floor(number_of_blks);
                for r in 0..num_round {
                    let sweep_size = 1 << r;
                    let number_of_workgroups = number_of_blks / (2 * sweep_size);

                    pass.set_push_constants(SWEEP_SIZE_OFFSET, bytemuck::bytes_of(&sweep_size));

                    dispatch_workgroup_ext(
                        &mut pass,
                        number_of_workgroups,
                        max_compute_workgroups_per_dimension,
                        WORKGROUP_OFFSET_OFFSET,
                    );
                }

                // scan down sweep(inclusive)
                pass.set_pipeline(scan_dnsweep_pipeline);
                let num_round = log2_ceil(number_of_blks).saturating_sub(1);
                for r in 0..num_round {
                    let num_slots = num_round - r;
                    let sweep_size = 1 << num_slots;

                    let num_src_blocks_with_full_slots = number_of_blks / (2 * sweep_size);
                    let extra_slots = 32 - (number_of_blks % sweep_size).leading_zeros();

                    let number_of_workgroups =
                        num_src_blocks_with_full_slots * num_slots + extra_slots;

                    pass.set_push_constants(SWEEP_SIZE_OFFSET, bytemuck::bytes_of(&sweep_size));

                    dispatch_workgroup_ext(
                        &mut pass,
                        number_of_workgroups,
                        max_compute_workgroups_per_dimension,
                        WORKGROUP_OFFSET_OFFSET,
                    );
                }

                // scan last block/histogram(exclusive)
                pass.set_pipeline(scan_last_block_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // scatter
            {
                pass.set_pipeline(scatter_pipeline);

                dispatch_workgroup_ext(
                    &mut pass,
                    number_of_blks,
                    max_compute_workgroups_per_dimension,
                    WORKGROUP_OFFSET_OFFSET,
                );
            }

            // Only the first pass needs to write the index to `global_vals_buf`
            pass.set_push_constants(INIT_INDEX_OFFSET, bytemuck::bytes_of(&0));
        }
    }
}

const fn log2_floor(x: u32) -> u32 {
    31 - x.leading_zeros()
}

const fn log2_ceil(x: u32) -> u32 {
    32 - x.leading_zeros() - (x.is_power_of_two() as u32)
}

pub fn dispatch_workgroup_ext(
    pass: &mut ComputePass,
    number_of_workgroups: u32,
    max_compute_workgroups_per_dimension: u32,
    workgroup_offset_offset: u32,
) {
    pass.set_push_constants(workgroup_offset_offset, bytemuck::bytes_of(&0));

    if number_of_workgroups <= max_compute_workgroups_per_dimension {
        pass.dispatch_workgroups(number_of_workgroups, 1, 1);
    } else {
        let d = number_of_workgroups / max_compute_workgroups_per_dimension;

        pass.dispatch_workgroups(max_compute_workgroups_per_dimension, d, 1);

        let workgroup_offset = max_compute_workgroups_per_dimension * d;
        pass.set_push_constants(
            workgroup_offset_offset,
            bytemuck::bytes_of(&workgroup_offset),
        );
        pass.dispatch_workgroups(number_of_workgroups - workgroup_offset, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use bevy::{
        render::{
            Render, RenderPlugin, RenderSet,
            render_resource::{
                Buffer, BufferAddress, BufferDescriptor, BufferInitDescriptor,
                CommandEncoderDescriptor, Maintain, MapMode,
            },
            renderer::RenderQueue,
        },
        scene::ScenePlugin,
    };

    use crate::GetSubgroupSizePlugin;

    use super::*;

    #[derive(Resource, Debug, Clone)]
    pub struct UnitTestHelper {
        /// The staging buffer used to store the input keys.
        ///
        /// cpu-buffer -> gpu-staging-buffer -> gpu-destination-buffer
        pub ikeys_staging_buf: Buffer,
        /// The staging buffer used to store the input vals.
        ///
        /// cpu-buffer -> gpu-staging-buffer -> gpu-destination-buffer
        pub ivals_staging_buf: Buffer,
        /// The staging buffer used to store the output keys.
        ///
        /// gpu-source-buffer -> gpu-staging-buffer -> cpu-buffer
        pub okeys_staging_buf: Buffer,
        /// The staging buffer used to store the output vals.
        ///
        /// gpu-source-buffer -> gpu-staging-buffer -> cpu-buffer
        pub ovals_staging_buf: Buffer,
    }

    impl UnitTestHelper {
        pub fn new(number_of_keys: u32, render_device: &RenderDevice) -> Self {
            let keys: Vec<u32> = (0..number_of_keys).rev().collect();
            let vals: Vec<u32> = (0..number_of_keys).collect();

            let ikeys_staging_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("radix_sort: input keys staging buffer"),
                usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
                contents: bytemuck::cast_slice(keys.as_slice()),
            });

            let ivals_staging_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("radix_sort: input vals staging buffer"),
                usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
                contents: bytemuck::cast_slice(vals.as_slice()),
            });

            let okeys_staging_buf = render_device.create_buffer(&BufferDescriptor {
                label: Some("radix_sort: output keys staging buffer"),
                size: (number_of_keys * NUMBER_OF_BYTES_PER_KEY) as BufferAddress,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let ovals_staging_buf = render_device.create_buffer(&BufferDescriptor {
                label: Some("radix_sort: output vals staging buffer"),
                size: (number_of_keys * NUMBER_OF_BYTES_PER_KEY) as BufferAddress,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            Self {
                ikeys_staging_buf,
                ivals_staging_buf,
                okeys_staging_buf,
                ovals_staging_buf,
            }
        }
    }

    fn prepare_unit_test_helper(
        mut commands: Commands,
        render_device: Res<RenderDevice>,
        radix_sort_settings: Res<RadixSortSettings>,
    ) {
        let unit_test_helper =
            UnitTestHelper::new(radix_sort_settings.max_number_of_keys(), &render_device);
        commands.insert_resource(unit_test_helper);
    }

    fn run_once(app: &mut App) {
        app.finish();
        app.cleanup();

        app.update();
    }

    fn create_unit_test_app(number_of_keys: u32) -> App {
        let mut app = App::new();

        app.add_plugins(MinimalPlugins)
            .add_plugins(WindowPlugin::default())
            .add_plugins(AssetPlugin::default())
            .add_plugins(ScenePlugin)
            .add_plugins(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .add_plugins(ImagePlugin::default())
            .add_plugins(GetSubgroupSizePlugin)
            .add_plugins(RadixSortPlugin {
                settings: number_of_keys.into(),
            });

        app.sub_app_mut(RenderApp).add_systems(
            Render,
            prepare_unit_test_helper
                .in_set(RenderSet::PrepareResources)
                .run_if(not(resource_exists::<UnitTestHelper>)),
        );

        app
    }

    fn run_radix_sort_test(
        number_of_keys: u32,
        pass_count: u32,
        is_sort_index: bool,
        read_from_even: bool,
    ) {
        let mut app = create_unit_test_app(number_of_keys);

        let unit_test_system =
            move |render_device: Res<RenderDevice>,
                  render_queue: Res<RenderQueue>,
                  pipeline_cache: Res<PipelineCache>,
                  radix_sort_pipeline: Res<RadixSortPipeline>,
                  radix_bind_group: Res<RadixSortBindGroup>,
                  unit_test_helper: Res<UnitTestHelper>,
                  sbufs: Res<RenderAssets<GpuShaderStorageBuffer>>| {
                let eve_global_keys_buf = sbufs
                    .get(EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
                    .unwrap();
                let eve_global_vals_buf = sbufs
                    .get(EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
                    .unwrap();
                let odd_global_keys_buf = sbufs
                    .get(ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
                    .unwrap();
                let odd_global_vals_buf = sbufs
                    .get(ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
                    .unwrap();

                let max_compute_workgroups_per_dimension =
                    render_device.limits().max_compute_workgroups_per_dimension;

                let copy_size = (number_of_keys * NUMBER_OF_BYTES_PER_KEY) as BufferAddress;

                let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("unit_test: radix_sort command encoder"),
                });

                let global_keys_buf = if read_from_even {
                    &eve_global_keys_buf.buffer
                } else {
                    &odd_global_keys_buf.buffer
                };
                encoder.copy_buffer_to_buffer(
                    &unit_test_helper.ikeys_staging_buf,
                    0,
                    global_keys_buf,
                    0,
                    copy_size,
                );

                if !is_sort_index {
                    let global_vals_buf = if read_from_even {
                        &eve_global_vals_buf.buffer
                    } else {
                        &odd_global_vals_buf.buffer
                    };
                    encoder.copy_buffer_to_buffer(
                        &unit_test_helper.ivals_staging_buf,
                        0,
                        global_vals_buf,
                        0,
                        copy_size,
                    );
                }

                run(
                    &mut encoder,
                    &pipeline_cache,
                    &radix_sort_pipeline,
                    &radix_bind_group,
                    max_compute_workgroups_per_dimension,
                    number_of_keys,
                    0..pass_count,
                    is_sort_index,
                    read_from_even,
                );

                let global_keys_buf = if (pass_count + read_from_even as u32) % 2 == 0 {
                    &odd_global_keys_buf.buffer
                } else {
                    &eve_global_keys_buf.buffer
                };
                encoder.copy_buffer_to_buffer(
                    global_keys_buf,
                    0,
                    &unit_test_helper.okeys_staging_buf,
                    0,
                    copy_size,
                );

                let global_vals_buf = if (pass_count + read_from_even as u32) % 2 == 0 {
                    &odd_global_vals_buf.buffer
                } else {
                    &eve_global_vals_buf.buffer
                };
                encoder.copy_buffer_to_buffer(
                    global_vals_buf,
                    0,
                    &unit_test_helper.ovals_staging_buf,
                    0,
                    copy_size,
                );

                render_queue.submit([encoder.finish()]);

                let size = (number_of_keys * NUMBER_OF_BYTES_PER_KEY) as BufferAddress;

                let keys_slice = unit_test_helper.okeys_staging_buf.slice(0..size);
                let vals_slice = unit_test_helper.ovals_staging_buf.slice(0..size);

                keys_slice.map_async(MapMode::Read, |_| ());
                vals_slice.map_async(MapMode::Read, |_| ());

                render_device.poll(Maintain::Wait).panic_on_timeout();

                // assert! sorted keys
                {
                    let view = keys_slice.get_mapped_range();
                    let data: &[u32] = bytemuck::cast_slice(&view);

                    // println!("");
                    // let nums = number_of_keys.div_ceil(NUMBER_OF_THREADS_PER_WORKGROUP);
                    // for i in 0..nums as usize {
                    //     let open = i * NUMBER_OF_THREADS_PER_WORKGROUP as usize;
                    //     let stop = (open + NUMBER_OF_THREADS_PER_WORKGROUP as usize)
                    //         .min(number_of_keys as usize);
                    //     let block = &data[open..stop];
                    //     println!("block {}: {:?}", i, block);
                    // }

                    let answer: Vec<u32> = (0..number_of_keys).collect();
                    assert_eq!(data, &answer);
                }

                // assert! sorted vals
                {
                    let view = vals_slice.get_mapped_range();
                    let data: &[u32] = bytemuck::cast_slice(&view);

                    let answer: Vec<u32> = (0..number_of_keys).rev().collect();
                    assert_eq!(data, &answer);
                }

                unit_test_helper.okeys_staging_buf.unmap();
                unit_test_helper.ovals_staging_buf.unmap();
            };

        app.sub_app_mut(RenderApp)
            .add_systems(Render, unit_test_system.in_set(RenderSet::Cleanup));

        run_once(&mut app);
    }

    #[test]
    fn test_rs_1() {
        run_radix_sort_test(1, 4, true, true);
        run_radix_sort_test(1, 3, false, true);
        run_radix_sort_test(1, 3, true, false);
    }

    #[test]
    fn test_rs_100() {
        run_radix_sort_test(100, 4, true, true);
        run_radix_sort_test(100, 3, false, true);
        run_radix_sort_test(100, 3, true, false);
    }

    #[test]
    fn test_rs_256() {
        run_radix_sort_test(256, 4, true, true);
        run_radix_sort_test(256, 3, false, true);
        run_radix_sort_test(256, 3, true, false);
    }

    #[test]
    fn test_rs_257() {
        run_radix_sort_test(257, 4, true, true);
        run_radix_sort_test(257, 3, false, true);
        run_radix_sort_test(257, 3, true, false);
    }

    #[test]
    fn test_rs_1000() {
        run_radix_sort_test(1_000, 4, true, true);
        run_radix_sort_test(1_000, 3, false, true);
        run_radix_sort_test(1_000, 3, true, false);
    }

    #[test]
    fn test_rs_16x256() {
        run_radix_sort_test(16 * 256, 4, true, true);
        run_radix_sort_test(16 * 256, 3, false, true);
        run_radix_sort_test(16 * 256, 3, true, false);
    }

    #[test]
    fn test_rs_16_000_000() {
        run_radix_sort_test(16_000_000, 4, true, true);
        run_radix_sort_test(16_000_000, 3, false, true);
        run_radix_sort_test(16_000_000, 3, true, false);
    }

    #[test]
    fn test_rs_16_777_216() {
        run_radix_sort_test(16_777_216, 4, true, true);
        run_radix_sort_test(16_777_216, 3, false, true);
        run_radix_sort_test(16_777_216, 3, true, false);
    }

    #[test]
    fn test_log2_floor() {
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(2), 1);

        assert_eq!(log2_floor(3), 1);
        assert_eq!(log2_floor(4), 2);
        assert_eq!(log2_floor(5), 2);

        assert_eq!(log2_floor(7), 2);
        assert_eq!(log2_floor(8), 3);
        assert_eq!(log2_floor(9), 3);

        assert_eq!(log2_floor(15), 3);
        assert_eq!(log2_floor(16), 4);
        assert_eq!(log2_floor(17), 4);
    }

    #[test]
    fn test_log2_ceil() {
        assert_eq!(log2_ceil(1), 0);
        assert_eq!(log2_ceil(2), 1);

        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(4), 2);
        assert_eq!(log2_ceil(5), 3);

        assert_eq!(log2_ceil(7), 3);
        assert_eq!(log2_ceil(8), 3);
        assert_eq!(log2_ceil(9), 4);

        assert_eq!(log2_ceil(15), 4);
        assert_eq!(log2_ceil(16), 4);
        assert_eq!(log2_ceil(17), 5);
    }
}
