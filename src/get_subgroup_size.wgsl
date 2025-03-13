@group(0) @binding(0) var<storage, read_write> g_subgroup_size: u32;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(subgroup_id) subgroup_id: u32, @builtin(subgroup_size) subgroup_size: u32) {
    if (subgroup_id == 0) {
        g_subgroup_size = subgroup_size;
    }
}