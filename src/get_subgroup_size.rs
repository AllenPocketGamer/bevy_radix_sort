use std::ops::Deref;

use bevy::{
    prelude::*,
    render::{
        RenderApp,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutEntries, Buffer, BufferDescriptor,
            BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
            Maintain, MapMode, PipelineLayoutDescriptor, RawComputePipelineDescriptor,
            ShaderModuleDescriptor, ShaderSource, ShaderStages, binding_types::storage_buffer,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};

pub const GET_SUBGROUPS_SIZE_SHADER: &str = include_str!("get_subgroup_size.wgsl");

pub struct GetSubgroupSizePlugin;

impl Plugin for GetSubgroupSizePlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        let render_device = render_app.world().resource::<RenderDevice>();
        let render_queue = render_app.world().resource::<RenderQueue>();

        let get_subgroup_size_utils = GetSubgroupSizeUtils::new(render_device);
        let subgroup_size = get_subgroup_size_utils.get_subgroup_size(render_device, render_queue);

        log::info!("subgroup_size: {}", subgroup_size.deref());

        render_app.insert_resource(subgroup_size);
        app.insert_resource(subgroup_size);
    }
}

#[derive(Resource, Debug, Clone)]
pub struct GetSubgroupSizeUtils {
    pipeline: ComputePipeline,
    buffer: Buffer,
    bind_group: BindGroup,
}

impl GetSubgroupSizeUtils {
    pub fn new(render_device: &RenderDevice) -> Self {
        let bind_group_layout = render_device.create_bind_group_layout(
            "get_subgroup_size bindgroup layout",
            &BindGroupLayoutEntries::single(ShaderStages::COMPUTE, storage_buffer::<u32>(false)),
        );

        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("get_subgroup_size buffer"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = render_device.create_bind_group(
            "get_subgroup bindgroup",
            &bind_group_layout,
            &BindGroupEntries::single(buffer.as_entire_binding()),
        );

        let shader = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("get_subgroup_size shader"),
            source: ShaderSource::Wgsl(GET_SUBGROUPS_SIZE_SHADER.into()),
        });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("get_subgroup_size pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("get_subgroup_size pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            buffer,
            bind_group,
        }
    }

    pub fn get_subgroup_size(
        &self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) -> SubgroupSize {
        let staging_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("get_subgroup_size staging buffer"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("get_subgroup_size command encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("get_subgroup_size pass"),
                ..default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buf, 0, 4);

        render_queue.submit([encoder.finish()]);

        {
            let slice = staging_buf.slice(0..4);
            slice.map_async(MapMode::Read, |_| ());
            render_device.poll(Maintain::Wait).panic_on_timeout();
            let subgroup_size: u32 = bytemuck::cast_slice(&slice.get_mapped_range())[0];
            staging_buf.unmap();

            SubgroupSize(subgroup_size)
        }
    }
}

#[derive(Resource, Debug, Clone, Copy, Deref)]
pub struct SubgroupSize(pub u32);

impl From<SubgroupSize> for u32 {
    fn from(value: SubgroupSize) -> Self {
        value.0
    }
}
impl From<&SubgroupSize> for u32 {
    fn from(value: &SubgroupSize) -> Self {
        value.0
    }
}
