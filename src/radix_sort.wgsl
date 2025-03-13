/// Read unsorted(sub-sort) keys from this buffer
@group(0) @binding(0) var<storage, read      > global_keys_i: array<u32>;
/// Read unsorted(sub-sort) vals from this buffer
@group(0) @binding(1) var<storage, read      > global_vals_i: array<u32>;
/// Read/Write histograms of count of each radix
@group(0) @binding(2) var<storage, read_write> global_blocks: array<u32>;
/// Write sorted(sub-sort) keys to this buffer
@group(0) @binding(3) var<storage, read_write> global_keys_o: array<u32>;
/// Write sorted(sub-sort) vals to this buffer
@group(0) @binding(4) var<storage, read_write> global_vals_o: array<u32>;

struct PushConstants {
    /// In most cases, the parameters `x`, `y`, `z` in [`ComputePass::dispatch_workgroups(x: u32, y: u32, z: u32)`]
    /// are limited to the range\[1, 65535\](the maximum number of workgroups per dimension can be queried through
    /// [`Limits::max_compute_workgroups_per_dimension`]).
    ///
    /// When dealing with a particularly large 1-dimensional array, for example, `number_of_keys` = 2^24,
    /// then `number_of_workgroups` = 2^16 = 65536; So (65536, 1, 1), (1, 65536, 1), and (1, 1, 65536) all exceed
    /// the valid range and will be rejected by the graphics API.
    ///
    /// Therefore, the simplest solution is to split one `dispatch_workgroups(..)` into two or more. Here I choose two:
    /// - `workgroup_offset` = 0:           dispatch_workgroups(65535, 1, 1)
    /// - `workgroup_offset` = 65535,       dispatch_workgroups(1, 1, 1)
    ///
    /// If `number_of_workgroups` is even larger, for example, `number_of_workgroups` = 2^24, then:
    /// - `workgroup_offset` = 0:           dispatch_workgroups(65535, 256, 1)
    /// - `workgroup_offset` = 16776960:    dispatch_workgroups(256, 1, 1)
    ///
    /// (Complaint: This is a very annoying limitation that adds unnecessary complexity to the code, but currently there is no better solution)
    workgroup_offset: u32,
    /// The number of keys to be sorted.
    number_of_keys: u32,
    /// The number of blocks(histogram) required.
    ///
    /// `number_of_blks` = ceil(`number_of_keys` / [`NUMBER_OF_THREADS_PER_WORKGROUP`])
    number_of_blks: u32,
    /// The current `pass` index being processed. For `u32` type with 8-bit `radix`, it requires 4 passes to process.
    /// So the valid range for `pass_index` is [0, 3].
    ///
    /// Since we are using the LSD (Least Significant Digit) sorting method, the `pass_index` represents:
    /// - `pass_index` = 0: Processing the least significant 8 bits of the `radix`,         0x000000XX
    /// - `pass_index` = 1: Processing the second least significant 8 bits of the `radix`,  0x0000XX00
    /// - `pass_index` = 2: Processing the second most significant 8 bits of the `radix`,   0x00XX0000
    /// - `pass_index` = 3: Processing the most significant 8 bits of the `radix`,          0xXX000000
    pass_index: u32,
    /// Used to control the step size of the prefix sum (inclusive) algorithm in step 2, up-sweep and down-sweep
    sweep_size: u32,
    /// Used to control whether to automatically write the index to `odd_global_vals_buf` in the 0th pass
    init_index: u32,
}
var<push_constant> pc: PushConstants;

const NUMBER_OF_KEYS_PER_SCATTER_BLOCK: u32 = #NUMBER_OF_THREADS_PER_WORKGROUP * #NUMBER_OF_ROWS_PER_WORKGROUP;

fn get_workgroup_index(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    return workgroup_id.y * num_workgroups.x + workgroup_id.x + pc.workgroup_offset;
}

fn get_radix_index(workgroup_index: u32, local_invocation_id_x: u32) -> u32 {
    return workgroup_index * #NUMBER_OF_THREADS_PER_WORKGROUP + local_invocation_id_x;
}

fn calc_radix(key: u32) -> u32 {
    return extractBits(key, pc.pass_index * #NUMBER_OF_RADIX_BITS, #{NUMBER_OF_RADIX_BITS}u);
}

#ifdef COUNT_RADIX_PIPELINE
var<workgroup> histogram: array<atomic<u32>, #NUMBER_OF_RADIX>;

@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u
) {
    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);
    let radix_index = get_radix_index(workgroup_index, local_invocation_id.x);

    // zeroing
    histogram[local_invocation_id.x] = 0u;

    workgroupBarrier();

    let start_index = workgroup_index * NUMBER_OF_KEYS_PER_SCATTER_BLOCK + local_invocation_id.x;
    let close_index = min(start_index + NUMBER_OF_KEYS_PER_SCATTER_BLOCK, pc.number_of_keys);
    for (var key_index = start_index; key_index < close_index; key_index += #{NUMBER_OF_THREADS_PER_WORKGROUP}u) {
        let key = global_keys_i[key_index];
        let radix = calc_radix(key);
        atomicAdd(&histogram[radix], 1u);
    }

    workgroupBarrier();

    global_blocks[radix_index] = histogram[local_invocation_id.x];
}
#endif // COUNT_RADIX_PIPELINE

#ifdef SCAN_UP_SWEEP_PIPELINE
@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u
) {
    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);

    let src_block_index = (2u * workgroup_index + 1u) * pc.sweep_size - 1u;
    let dst_block_index = src_block_index + pc.sweep_size;

    let src_radix_count_index = get_radix_index(src_block_index, local_invocation_id.x);
    let dst_radix_count_index = get_radix_index(dst_block_index, local_invocation_id.x);
    global_blocks[dst_radix_count_index] += global_blocks[src_radix_count_index];
}
#endif // SCAN_UP_SWEEP_PIPELINE

#ifdef SCAN_DOWN_SWEEP_PIPELINE
fn ulog2(x: u32) -> u32 {
    return 31u - countLeadingZeros(x);
}

@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
) {
    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);
    let num_slots = ulog2(pc.sweep_size);

    let src_block_id = workgroup_index / num_slots;
    let dst_block_id = workgroup_index % num_slots;

    let src_block_index = (2u * src_block_id + 1u) * pc.sweep_size - 1u;
    let dst_block_index = src_block_index + (1u << dst_block_id);

    let src_radix_count_index = get_radix_index(src_block_index, local_invocation_id.x);
    let dst_radix_count_index = get_radix_index(dst_block_index, local_invocation_id.x);
    global_blocks[dst_radix_count_index] += global_blocks[src_radix_count_index];
}
#endif // SCAN_DOWN_SWEEP_PIPELINE

#ifdef SCAN_LAST_BLOCK_PIPELINE
const NUMBER_OF_SUBGROUPS: u32 = #NUMBER_OF_THREADS_PER_WORKGROUP / #NUMBER_OF_THREADS_PER_SUBGROUP;

var<workgroup> subgroup_sums: array<u32, NUMBER_OF_SUBGROUPS>;

fn scan_exclusive(value: u32, subgroup_id: u32, subgroup_invocation_id: u32) -> u32 {
    let subgroup_prefix_sum = subgroupInclusiveAdd(value);

    if subgroup_invocation_id == #NUMBER_OF_THREADS_PER_SUBGROUP - 1u { subgroup_sums[subgroup_id] = subgroup_prefix_sum; }
    workgroupBarrier();
    
    let prev_subgroup_sum = select(0u, subgroup_sums[subgroup_invocation_id], subgroup_invocation_id < subgroup_id);
    let prev_sum = subgroupAdd(prev_subgroup_sum);

    return prev_sum + subgroup_prefix_sum - value;
}

@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    let block_index = pc.number_of_blks - 1u;
    let radix_count_index = get_radix_index(block_index, local_invocation_id.x);
    let radix_count = global_blocks[radix_count_index];

    let prefix_sum_exclusive = scan_exclusive(radix_count, subgroup_id, subgroup_invocation_id);

    global_blocks[radix_count_index] = prefix_sum_exclusive;
}
#endif // SCAN_LAST_BLOCK_PIPELINE

#ifdef SCATTER_PIPELINE
const NUMBER_OF_SUBGROUPS: u32 = #NUMBER_OF_THREADS_PER_WORKGROUP / #NUMBER_OF_THREADS_PER_SUBGROUP;
const NUMBER_OF_RADIX_COUNTS: u32 = #NUMBER_OF_RADIX * NUMBER_OF_SUBGROUPS;

// In the first stage, `subgroup_histograms` is used as a histogram for counting `radix` within the `subgroup`:
//
// If [`NUMBER_OF_SUBGROUPS`] = 8, the memory layout of `subgroup_histograms` is as follows:
//                                                                                                            
//            rdx0  rdx1  rdx2  rdx3   ...  rdx255 
//           +-----+-----+-----+-----+-----+-----+ 
// subgroup0 | c0  | c1  | c2  | c3  | ... |c255 | 
//           +-----+-----+-----+-----+-----+-----+ 
//       ... | ... | ... | ... | ... | ... | ... | 
//           +-----+-----+-----+-----+-----+-----+ 
// subgroup7 |c1792|c1793|c1794|c1795| ... |c2047| 
//           +-----+-----+-----+-----+-----+-----+ 
//
// In the second stage, `subgroup_histograms` is used as an auxiliary container for reordering the `SCATTER_BLOCK`.
var<workgroup> subgroup_histograms: array<u32, max(NUMBER_OF_KEYS_PER_SCATTER_BLOCK, NUMBER_OF_RADIX_COUNTS)>;
// A histogram stores the `local_radix_offset`/`global_radix_offset`
var<workgroup> histogram: array<u32, #NUMBER_OF_RADIX>;

// 1. Each thread will load the corresponding column data (keys/vals) in the `SCATTER_BLOCK`;
// 2. The `SCATTER_BLOCK` will be sorted, and the sorted results will be written back into `thread_keys/thread_vals` in row order;
// 3. The data in `thread_keys/thread_vals` will be written into `global_keys_o/global_vals_o` according to `global_radix_offset`;
//
// ## Why use such a complex data structure?
//
// It is to avoid delays caused by high `L2 Cache Throughput`.
var<private> thread_keys: array<u32, #NUMBER_OF_ROWS_PER_WORKGROUP>;
var<private> thread_vals: array<u32, #NUMBER_OF_ROWS_PER_WORKGROUP>;
// `order` indicates the number of times the corresponding key's radix appears in the `SCATTER_BLOCK`
var<private> thread_ords: array<u32, #NUMBER_OF_ROWS_PER_WORKGROUP>;

// In the `scatter` step, when using `scan_exclusive`, `subgroup_histograms` is idle and can be used as `subgroup_sums`,
// saving the use of `shared memory` (although it's not much).
fn scan_exclusive(value: u32, subgroup_id: u32, subgroup_invocation_id: u32) -> u32 {
    let subgroup_prefix_sum = subgroupInclusiveAdd(value);

    if subgroup_invocation_id == #NUMBER_OF_THREADS_PER_SUBGROUP - 1u { subgroup_histograms[subgroup_id] = subgroup_prefix_sum; }
    workgroupBarrier();
    
    let prev_subgroup_sum = select(0u, subgroup_histograms[subgroup_invocation_id], subgroup_invocation_id < subgroup_id);
    let prev_sum = subgroupAdd(prev_subgroup_sum);

    return prev_sum + subgroup_prefix_sum - value;
}

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

fn fill_global_radix_offset(workgroup_index: u32, local_invocation_id_x: u32) {
    let last_block_index = pc.number_of_blks - 1u;
    let radix_initial_offset_index = get_radix_index(last_block_index, local_invocation_id_x);

    var radix_offset = global_blocks[radix_initial_offset_index];

    if workgroup_index > 0u {
        let curr_block_index = workgroup_index - 1u;
        let radix_internal_offset_index = get_radix_index(curr_block_index, local_invocation_id_x);
        radix_offset += global_blocks[radix_internal_offset_index];
    }

    histogram[local_invocation_id_x] = radix_offset;
}

fn count_one_bits_vec4u(mask: vec4u) -> u32 {
    let counts = countOneBits(mask);
    return counts.x + counts.y + counts.z + counts.w;
}

// If:
// - `sgtid` = 0 , return vec4u(b0000_0000_0000_0000_0000_0000_0000_0000, 0, 0, 0)
// - `sgtid` = 16, return vec4u(b0000_0000_0000_0000_1111_1111_1111_1111, 0, 0, 0)
// - `sgtid` = 31, return vec4u(b0111_1111_1111_1111_1111_1111_1111_1111, 0, 0, 0)
// - `sgtid` = 33, return vec4u(b1111_1111_1111_1111_1111_1111_1111_1111, b0000_0000_0000_0000_0000_0000_0000_0001..., 0, 0)
// - ...
fn calc_prv_sgtid_subgroup_mask(sgtid: u32) -> vec4u {
    // The compiler will automatically constant-fold
    let number_of_u32_bits = 32u;
    let base_offset = vec4u(0, 1, 2, 3) * number_of_u32_bits;
    let mask_all = vec4u(0xFFFFFFFFu);

    let offset = max(vec4u(sgtid), base_offset) - base_offset;
    return select(mask_all, (vec4u(1) << offset) - 1u, offset < vec4u(number_of_u32_bits));
}

@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);

    // zeroing: no workgroupBarrier() required
    histogram[local_invocation_id.x] = 0u;

    let base_index = workgroup_index * NUMBER_OF_KEYS_PER_SCATTER_BLOCK;
    let number_of_keys_of_scatter_block = min(NUMBER_OF_KEYS_PER_SCATTER_BLOCK, pc.number_of_keys - base_index);
    let number_of_rows_of_scatter_block = div_ceil(number_of_keys_of_scatter_block, #{NUMBER_OF_THREADS_PER_WORKGROUP}u);

    var key_index = base_index + local_invocation_id.x;
    for (var row = 0u; row < number_of_rows_of_scatter_block; row++) {
        let is_active = key_index < pc.number_of_keys;

        // Avoid reading out-of-bounds data
        var key = 0xFFFFFFFFu;
        var val = key_index;
        if is_active {
            key = global_keys_i[key_index];
            if pc.init_index == 0u { val = global_vals_i[key_index]; }
        }
        
        let radix = calc_radix(key);

        // This loop's task is to find all `subgroup_threads` in the `subgroup` that have the same `radix`.
        //
        // ## Idea
        //
        // `radix` is 8 bits, iterate from low to high for each bit:
        //  1. Use `subgroupBallot(..)` to get the values of other `subgroup_threads` in the `subgroup` at this bit,
        //      denoted as: `radix_1bit_subgroup_mask`;
        //  2. Use 1 to represent all `subgroup_threads` that have the same bit value as this thread
        //      (if the bit is 0, perform a bitwise NOT operation on `radix_1bit_subgroup_mask`);
        //  3. `radix_subgroup_mask` &= `radix_1bit_subgroup_mask`
        //
        // After the loop, each bit in `radix_subgroup_mask` represents a `subgroup_thread`,
        // and these `subgroup_threads` have the same radix as this thread.
        //
        // `radix_subgroup_mask` is initialized with `subgroupBallot(is_active)`, 
        // which is a mask where each bit represents an active `subgroup_thread`.
        var radix_subgroup_mask = subgroupBallot(is_active);
        for (var i = 0u; i < #NUMBER_OF_RADIX_BITS; i++) {
            let radix_1bit = extractBits(radix, i, 1u);
            //                       +-----+-----+-----+-----+-----+    
            //                 sgtid | 31  | 30  | 29  | ... |  0  |    
            //                       +-----+-----+-----+-----+-----+    
            // radix_1bit_sgtid_mask |  x  |  x  |  x  | ... |  x  |    
            //                       +-----+-----+-----+-----+-----+  
            let radix_1bit_subgroup_mask = subgroupBallot(bool(radix_1bit));

            radix_subgroup_mask &= select(radix_1bit_subgroup_mask, ~radix_1bit_subgroup_mask, radix_1bit == 0u);
        }

        // The number of the radix has appeared before this subgroup_invocation_id in the subgroup.
        let prv_sgtid_subgroup_mask = calc_prv_sgtid_subgroup_mask(subgroup_invocation_id);
        let prv_sgtid_radix_count_of_subgroup = count_one_bits_vec4u(prv_sgtid_subgroup_mask & radix_subgroup_mask);

        // zeroing: no workgroupBarrier() required
        let base_index = subgroup_id * #NUMBER_OF_RADIX + subgroup_invocation_id;
        let close_index = subgroup_id * #NUMBER_OF_RADIX + #NUMBER_OF_RADIX;
        for (var i = base_index; i < close_index; i += #{NUMBER_OF_THREADS_PER_SUBGROUP}u) {
            subgroup_histograms[i] = 0u;
        }

        let radix_count_index = subgroup_id * #NUMBER_OF_RADIX + radix;
        subgroup_histograms[radix_count_index] = count_one_bits_vec4u(radix_subgroup_mask);

        workgroupBarrier();

        let radix_accumulation = histogram[radix];

        // prefix sum exclusively
        var accumulation = 0u;
        for (var i = local_invocation_id.x; i < NUMBER_OF_RADIX_COUNTS; i += #{NUMBER_OF_RADIX}u) {
            let radix_count_of_subgroup = subgroup_histograms[i];
            subgroup_histograms[i] = accumulation;
            accumulation += radix_count_of_subgroup;
        }

        workgroupBarrier();

        histogram[local_invocation_id.x] += accumulation;

        // Don't worry about writing out-of-bounds key/val values, subsequent steps will filter them out
        // No Need: if is_active { ... }
        thread_keys[row] = key;
        thread_vals[row] = val;
        thread_ords[row] = radix_accumulation + subgroup_histograms[radix_count_index] + prv_sgtid_radix_count_of_subgroup;

        key_index += #{NUMBER_OF_THREADS_PER_WORKGROUP}u;
    }

    workgroupBarrier();

    // Calculate the local_radix_offset
    histogram[local_invocation_id.x] = scan_exclusive(histogram[local_invocation_id.x], subgroup_id, subgroup_invocation_id);

    workgroupBarrier();

    // Add local_ordered_index to high 16-bit
    for (var row = 0u; row < number_of_rows_of_scatter_block; row++) {
        let key = thread_keys[row];
        let ord = thread_ords[row];

        let radix = calc_radix(key);
        let local_ordered_index = histogram[radix] + ord;

        thread_ords[row] = (local_ordered_index << 16u) | ord;
    }

    workgroupBarrier();

    // `local_radix_offset` stored in `histogram` is not useful anymore, so we can reuse it to store `global_radix_offset`
    fill_global_radix_offset(workgroup_index, local_invocation_id.x);

    workgroupBarrier();

    // KEYS: reorder in the `SCATTER_BLOCK`
    for (var row = 0u; row < number_of_rows_of_scatter_block; row++) {
        let key = thread_keys[row];
        // NOTE: Performance issue
        //
        // This is very strange. When reading data from `thread_ords`, (observed through `NSight Graphics`) the active warps of the SM
        // drop sharply to 2/3 of the original,
        // and the overhead increases from 0.3ms to 0.5ms (4070Tis).
        //
        // Also, it's unclear why the VRAM Throughput is only around 50%, theoretically it should be close to 85%.
        //
        // These two bottlenecks limit the overall performance, but I don't know how to solve them, 
        // nor do I know if there is a problem with my code, so I'll leave it for now.
        let local_ordered_index = thread_ords[row] >> 16u;

        subgroup_histograms[local_ordered_index] = key;
    }

    workgroupBarrier();

    // KEYS: write the sorted back to the `thread_keys`
    key_index = local_invocation_id.x;
    for (var row = 0u; row < number_of_rows_of_scatter_block; row++) {
        thread_keys[row] = subgroup_histograms[key_index];

        key_index += #{NUMBER_OF_THREADS_PER_WORKGROUP}u;
    }

    workgroupBarrier();

    // VALS: reorder in the `SCATTER_BLOCK`
    for (var row = 0u; row < number_of_rows_of_scatter_block; row++) {
        let val = thread_vals[row];
        let local_ordered_index = thread_ords[row] >> 16u;

        subgroup_histograms[local_ordered_index] = val;
    }

    workgroupBarrier();

    // VALS: write the sorted back to the `thread_vals`
    key_index = local_invocation_id.x;
    for (var row = 0u; row < number_of_rows_of_scatter_block; row++) {
        thread_vals[row] = subgroup_histograms[key_index];

        key_index += #{NUMBER_OF_THREADS_PER_WORKGROUP}u;
    }

    workgroupBarrier();

    // ORDS: reoder in the `SCATTER_BLOCK`
    for (var row = 0u; row < number_of_rows_of_scatter_block; row++) {
        let ord = thread_ords[row];
        let local_ordered_index = thread_ords[row] >> 16u;

        // Remove the high 16-bit
        subgroup_histograms[local_ordered_index] = ord & 0xFFFFu;
    }

    workgroupBarrier();

    // ORDS: write the sorted back to the `thread_ords`
    key_index = local_invocation_id.x;
    for (var row = 0u; row < number_of_rows_of_scatter_block; row++) {
        thread_ords[row] = subgroup_histograms[key_index];

        key_index += #{NUMBER_OF_THREADS_PER_WORKGROUP}u;
    }

    // Write the sorted results back to the `global_keys_o/global_vals_o`
    key_index = base_index + local_invocation_id.x;
    for (var row = 0u; row < number_of_rows_of_scatter_block; row++) {
        let is_active = key_index < pc.number_of_keys;

        if is_active {
            let key = thread_keys[row];
            let val = thread_vals[row];
            let ord = thread_ords[row];

            let radix = calc_radix(key);

            let global_ordered_index = histogram[radix] + ord;

            global_keys_o[global_ordered_index] = key;
            global_vals_o[global_ordered_index] = val;
        }

        key_index += #{NUMBER_OF_THREADS_PER_WORKGROUP}u;
    }
}
#endif // SCATTER_PIPELINE