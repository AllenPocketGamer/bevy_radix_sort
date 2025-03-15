use bevy::{
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderSet,
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            Buffer, BufferAddress, BufferDescriptor, BufferUsages, Maintain, MapMode, PipelineCache,
        },
        renderer::{RenderContext, RenderDevice},
        storage::GpuShaderStorageBuffer,
    },
};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use bevy_radix_sort::{
    EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE,
    GetSubgroupSizePlugin, LoadState, RadixSortBindGroup, RadixSortPipeline, RadixSortPlugin,
    RadixSortSettings,
};
use rand::Rng;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_plugins(SimpleGpuSortPlugin {
            max_number_of_keys: 1024 * 1024,
        })
        .add_event::<SortEvent>()
        .run();
}

// Default length of random keys to sort
const DEFAULT_RANDOM_KEYS_LEN: usize = 256;

#[derive(Event, Default, Clone, Copy)]
pub struct SortEvent {
    pub length: usize,
}

#[derive(Resource, Default, Clone, Copy)]
pub struct SortCommand {
    pub requested: bool,
    pub length: usize,
}

#[derive(Resource, Default)]
pub struct InputState {
    pub input_value: String,
    pub last_sorted_length: Option<usize>,
    pub continuous_generation: bool,
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
            .init_resource::<SortCommand>()
            .init_resource::<InputState>()
            .add_systems(
                Update,
                (disable_requested, ui_system, handle_sort_event).chain(),
            );

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<InputState>()
            .add_systems(ExtractSchedule, extract_sort_command)
            .add_systems(
                Render,
                SimpleGpuSortResource::initialize
                    .in_set(RenderSet::PrepareResources)
                    .run_if(not(resource_exists::<SimpleGpuSortResource>)),
            )
            .add_systems(
                Render,
                write_radom_kvs_to_staging_bufs
                    .in_set(RenderSet::PrepareResources)
                    .run_if(resource_exists::<SimpleGpuSortResource>),
            )
            .add_systems(
                Render,
                read_sorted_kvs_from_gpu_storage_bufs.after(RenderSet::Render),
            );

        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(SimpleGpuSortNodeLabel, SimpleGpuSortNode::default());
        graph.add_node_edge(
            bevy::render::graph::CameraDriverLabel,
            SimpleGpuSortNodeLabel,
        );
    }
}

// Egui UI system
fn ui_system(
    mut contexts: EguiContexts,
    mut input_state: ResMut<InputState>,
    mut sort_events: EventWriter<SortEvent>,
) {
    egui::Window::new("Radix Sort Control")
        .default_pos([100.0, 100.0])
        .default_size([300.0, 250.0])
        .show(contexts.ctx_mut(), |ui| {
            ui.heading("Bevy Radix Sort");

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(10.0);

            ui.label("Array Length:");

            // Input field for array length
            let mut input_value = input_state.input_value.clone();
            if input_value.is_empty() {
                input_value = DEFAULT_RANDOM_KEYS_LEN.to_string();
            }

            let response = ui.add(egui::TextEdit::singleline(&mut input_value)
                .hint_text(DEFAULT_RANDOM_KEYS_LEN.to_string())
                .desired_width(120.0));

            if response.changed() {
                // Filter out non-numeric characters
                input_state.input_value = input_value
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect();
            }

            // Continuous Generation checkbox
            ui.checkbox(&mut input_state.continuous_generation, "Continuous Generation");
            if input_state.continuous_generation {
                ui.label("Automatically generates and sorts new random kvs");

                let length = if input_state.input_value.is_empty() {
                    DEFAULT_RANDOM_KEYS_LEN
                } else {
                    input_state.input_value.parse().unwrap_or(DEFAULT_RANDOM_KEYS_LEN)
                };

                sort_events.send(SortEvent { length });
            }

            ui.add_space(10.0);

            // Sort button (disabled if continuous generation is enabled)
            ui.horizontal(|ui| {
                let button = ui.add_enabled(!input_state.continuous_generation,
                                           egui::Button::new("Sort Data"));

                if button.clicked() {
                    // Parse the input value, or use the default if empty or invalid
                    let length = if input_state.input_value.is_empty() {
                        DEFAULT_RANDOM_KEYS_LEN
                    } else {
                        input_state.input_value.parse().unwrap_or(DEFAULT_RANDOM_KEYS_LEN)
                    };

                    // Send a SortEvent when the button is pressed
                    sort_events.send(SortEvent { length });
                    input_state.last_sorted_length = Some(length);

                    info!("Sort button pressed, sorting {} elements", length);
                }
            });

            ui.add_space(5.0);

            // Information about the sort
            ui.collapsing("About Radix Sort", |ui| {
                ui.label("Radix sort is a non-comparative sorting algorithm that sorts data with integer keys by grouping keys by individual digits.");
                ui.label("This implementation uses the GPU to perform the sort, making it very fast for large arrays.");
                ui.label("The maximum array size is determined by the GPU memory.");
            });
        });
}

fn disable_requested(mut sort_command: ResMut<SortCommand>) {
    sort_command.requested = false;
}

fn handle_sort_event(mut events: EventReader<SortEvent>, mut sort_command: ResMut<SortCommand>) {
    for event in events.read() {
        sort_command.requested = true;
        sort_command.length = event.length;
    }
}

fn extract_sort_command(mut commands: Commands, sort_command: Extract<Res<SortCommand>>) {
    commands.insert_resource(SortCommand {
        requested: sort_command.requested,
        length: sort_command.length,
    });
}

fn write_radom_kvs_to_staging_bufs(
    mut sort_resource: ResMut<SimpleGpuSortResource>,
    device: Res<RenderDevice>,
    sort_command: Res<SortCommand>,
) {
    if sort_command.requested {
        sort_resource.write_random_kvs_to_staging_bufs(&device, sort_command.length);
    }
}

fn read_sorted_kvs_from_gpu_storage_bufs(
    mut sort_resource: ResMut<SimpleGpuSortResource>,
    device: Res<RenderDevice>,
    sort_command: Res<SortCommand>,
) {
    if sort_command.requested {
        sort_resource.read_sorted_kvs_from_gpu_storage_bufs(&device);
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
        });
    }

    pub fn write_random_kvs_to_staging_bufs(&mut self, device: &RenderDevice, length: usize) {
        let mut rng = rand::thread_rng();
        let keys: Vec<u32> = (0..length)
            .map(|_| rng.gen_range(0..length as u32))
            .collect();
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

        self.length = length;

        info!("Generated {} random keys", length);
        info!("Keys: {:?}", &keys);
        info!("Vals: {:?}", &vals);
    }

    pub fn read_sorted_kvs_from_gpu_storage_bufs(&mut self, device: &RenderDevice) {
        let size = (self.length * std::mem::size_of::<u32>()) as BufferAddress;
        let keys_slice = self.o_keys_buf.slice(0..size);
        let vals_slice = self.o_vals_buf.slice(0..size);

        keys_slice.map_async(MapMode::Read, |_| ());
        vals_slice.map_async(MapMode::Read, |_| ());

        device.poll(Maintain::wait()).panic_on_timeout();

        {
            let keys_view = keys_slice.get_mapped_range();
            let keys: &[u32] = bytemuck::cast_slice(&keys_view);
            let vals_view = vals_slice.get_mapped_range();
            let vals: &[u32] = bytemuck::cast_slice(&vals_view);

            info!("Sorted {} random keys", self.length);
            info!("Keys: {:?}", &keys);
            info!("Vals: {:?}", &vals);
        }

        self.o_keys_buf.unmap();
        self.o_vals_buf.unmap();
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
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if matches!(self.state, SimpleGpuSortState::OnLoad)
            || !world.resource::<SortCommand>().requested
        {
            return Ok(());
        }

        let max_compute_workgroups_per_dimension = {
            let render_device = world.resource::<RenderDevice>();
            render_device.limits().max_compute_workgroups_per_dimension
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let radix_sort_pipeline = world.resource::<RadixSortPipeline>();
        let radix_sort_bind_group = world.resource::<RadixSortBindGroup>();

        let simple_gpu_sort_resource = world.resource::<SimpleGpuSortResource>();

        let storage_buffers = world.resource::<RenderAssets<GpuShaderStorageBuffer>>();
        let eve_global_keys_buf = storage_buffers
            .get(EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let eve_global_vals_buf = storage_buffers
            .get(EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();

        let encoder = render_context.command_encoder();

        let size = (simple_gpu_sort_resource.length * std::mem::size_of::<u32>()) as BufferAddress;

        encoder.copy_buffer_to_buffer(
            &simple_gpu_sort_resource.i_keys_buf,
            0,
            &eve_global_keys_buf.buffer,
            0,
            size,
        );

        encoder.copy_buffer_to_buffer(
            &simple_gpu_sort_resource.i_vals_buf,
            0,
            &eve_global_vals_buf.buffer,
            0,
            size,
        );

        info!("before radix_sort: copy key/val from staging buffer to gpu storage buffer");

        bevy_radix_sort::run(
            encoder,
            pipeline_cache,
            radix_sort_pipeline,
            radix_sort_bind_group,
            max_compute_workgroups_per_dimension,
            simple_gpu_sort_resource.length as u32,
            0..4,
            false,
            true,
        );

        encoder.copy_buffer_to_buffer(
            &eve_global_keys_buf.buffer,
            0,
            &simple_gpu_sort_resource.o_keys_buf,
            0,
            size,
        );

        encoder.copy_buffer_to_buffer(
            &eve_global_vals_buf.buffer,
            0,
            &simple_gpu_sort_resource.o_vals_buf,
            0,
            size,
        );

        info!("after radix_sort: copy key/val from gpu storage buffer to staging buffer");

        Ok(())
    }
}
