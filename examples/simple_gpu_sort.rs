use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        render_graph::{self, RenderLabel},
        render_resource::{
            Buffer, BufferAddress, BufferDescriptor, BufferUsages, Extent3d, Maintain, MapMode,
            PipelineCache, TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice},
    },
    window::WindowResolution,
};
use bevy_radix_sort::{
    GetSubgroupSizePlugin, LoadState, RadixSortBindGroup, RadixSortPipeline, RadixSortPlugin,
    RadixSortSettings,
};
use rand::Rng;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SimpleGpuSortPlugin {
            max_number_of_keys: 1024 * 1024,
        })
        .run();
}

#[derive(Debug)]
pub struct SimpleGpuSortPlugin {
    pub max_number_of_keys: u32,
}

impl Plugin for SimpleGpuSortPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(GetSubgroupSizePlugin)
            .add_plugins(RadixSortPlugin {
                settings: self.max_number_of_keys.into(),
            })
            .sub_app_mut(RenderApp)
            .add_systems(
                Render,
                SimpleGpuSortResource::initialize
                    .in_set(RenderSet::PrepareResources)
                    .run_if(not(resource_exists::<SimpleGpuSortResource>)),
            );
    }
}

#[derive(Resource)]
pub struct SimpleGpuSortResource {
    // cpu-buffer -> gpu-staging-buffer -> gpu-destination-buffer
    pub i_keys_buf: Buffer,
    pub i_vals_buf: Buffer,
    // gpu-source-buffer -> gpu-staging-buffer -> cpu-buffer
    pub o_keys_buf: Buffer,
    pub o_vals_buf: Buffer,
    // the length of the keys and vals
    pub length: usize,
    // is the sort done?
    pub sorted: bool,
}

impl SimpleGpuSortResource {
    pub fn initialize(
        mut commands: Commands,
        device: Res<RenderDevice>,
        radix_sort_settings: Res<RadixSortSettings>,
    ) {
        let max_number_of_keys = radix_sort_settings.max_number_of_keys() as usize;

        let i_keys_buf = device.create_buffer(&BufferDescriptor {
            label: Some("copy keys from cpu to gpu"),
            size: (max_number_of_keys * std::mem::size_of::<u32>()) as BufferAddress,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        let i_vals_buf = device.create_buffer(&BufferDescriptor {
            label: Some("copy vals from cpu to gpu"),
            size: (max_number_of_keys * std::mem::size_of::<u32>()) as BufferAddress,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        let o_keys_buf = device.create_buffer(&BufferDescriptor {
            label: Some("copy keys from gpu to cpu"),
            size: (max_number_of_keys * std::mem::size_of::<u32>()) as BufferAddress,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let o_vals_buf = device.create_buffer(&BufferDescriptor {
            label: Some("copy vals from gpu to cpu"),
            size: (max_number_of_keys * std::mem::size_of::<u32>()) as BufferAddress,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        commands.insert_resource(Self {
            i_keys_buf,
            i_vals_buf,
            o_keys_buf,
            o_vals_buf,
            length: max_number_of_keys,
            sorted: true,
        });
    }

    pub fn generate_random_keys(&mut self, device: &RenderDevice, length: usize) {
        let mut rng = rand::thread_rng();
        let keys: Vec<u32> = (0..length).map(|_| rng.gen_range(0..u32::MAX)).collect();
        let vals: Vec<u32> = (0u32..length as u32).collect();

        let size = (length * std::mem::size_of::<u32>()) as BufferAddress;
        let keys_slice = self.i_keys_buf.slice(0..size);
        let vals_slice = self.i_vals_buf.slice(0..size);

        keys_slice.map_async(MapMode::Write, |_| ());
        vals_slice.map_async(MapMode::Write, |_| ());

        device.poll(Maintain::wait()).panic_on_timeout();

        keys_slice.get_mapped_range_mut()[..size as usize]
            .copy_from_slice(bytemuck::cast_slice(&keys));
        vals_slice.get_mapped_range_mut()[..size as usize]
            .copy_from_slice(bytemuck::cast_slice(&vals));

        self.i_keys_buf.unmap();
        self.i_vals_buf.unmap();
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, RenderLabel)]
pub struct SimpleGpuSortNodeLabel;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum SimpleGpuSortState {
    #[default]
    OnLoad,
    Loaded,
}

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct SimpleGpuSortNode {
    state: SimpleGpuSortState,
}

impl render_graph::Node for SimpleGpuSortNode {
    fn update(&mut self, world: &mut World) {
        if matches!(self.state, SimpleGpuSortState::OnLoad) {
            let radix_sort_load_state = bevy_radix_sort::check_load_state(world);

            if let LoadState::Failed(err) = &radix_sort_load_state {
                panic!("{}", err);
            }

            if matches!(radix_sort_load_state, LoadState::Loaded) {
                self.state = SimpleGpuSortState::Loaded;
            }
        }
    }

    fn run(
        &self,
        graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if matches!(self.state, SimpleGpuSortState::OnLoad) {
            return Ok(());
        }

        // let max_compute_workgroups_per_dimension = {
        //     let render_device = world.resource::<RenderDevice>();
        //     render_device.limits().max_compute_workgroups_per_dimension
        // };

        // let pipeline_cache = world.resource::<PipelineCache>();

        // let radix_sort_pipeline = world.resource::<RadixSortPipeline>();
        // let radix_sort_bind_group = world.resource::<RadixSortBindGroup>();

        // let encoder = render_context.command_encoder();

        // bevy_radix_sort::run(
        //     encoder,
        //     pipeline_cache,
        //     radix_sort_pipeline,
        //     radix_sort_bind_group,
        //     max_compute_workgroups_per_dimension,
        //     todo!(),
        //     0..4,
        //     true,
        //     true,
        // );

        Ok(())
    }
}
