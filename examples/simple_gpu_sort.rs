use bevy::{
    color::palettes::css::RED,
    core_pipeline::core_2d::graph::{Core2d, Node2d},
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
use bevy_radix_sort::{
    EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE,
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
        .add_event::<SortEvent>()
        .run();
}

// Constants for button colors
const NORMAL_BUTTON: Color = Color::srgb(0.15, 0.15, 0.15);
const HOVERED_BUTTON: Color = Color::srgb(0.25, 0.25, 0.25);
const PRESSED_BUTTON: Color = Color::srgb(0.35, 0.75, 0.35);

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
            .add_systems(Startup, setup_ui)
            .add_systems(
                Update,
                (disable_requested, handle_sort_event, button_system).chain(),
            );

        let render_app = app.sub_app_mut(RenderApp);

        render_app
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

        if let Some(core2d) = render_app
            .world_mut()
            .resource_mut::<RenderGraph>()
            .get_sub_graph_mut(Core2d)
        {
            core2d.add_node(SimpleGpuSortNodeLabel, SimpleGpuSortNode::default());
            core2d.add_node_edge(Node2d::EndMainPassPostProcessing, SimpleGpuSortNodeLabel);
        }
    }
}

// Setup UI system to create the button
fn setup_ui(mut commands: Commands) {
    // UI camera
    commands.spawn(Camera2d);

    // UI root node
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        })
        .with_children(|parent| {
            // Sort button
            parent
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(200.0),
                        height: Val::Px(65.0),
                        border: UiRect::all(Val::Px(5.0)),
                        // horizontally center child text
                        justify_content: JustifyContent::Center,
                        // vertically center child text
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BorderColor(Color::BLACK),
                    BackgroundColor(NORMAL_BUTTON),
                ))
                .with_child((
                    Text::new("radix_sort"),
                    TextFont {
                        font_size: 30.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.9, 0.9, 0.9)),
                ));
        });
}

// Button interaction system
fn button_system(
    mut interaction_query: Query<
        (
            &Interaction,
            &mut BackgroundColor,
            &mut BorderColor,
            &Children,
        ),
        (Changed<Interaction>, With<Button>),
    >,
    mut sort_events: EventWriter<SortEvent>,
) {
    for (interaction, mut color, mut border_color, _) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = PRESSED_BUTTON.into();
                border_color.0 = RED.into();

                // Send a SortEvent when the button is pressed
                sort_events.send(SortEvent {
                    length: DEFAULT_RANDOM_KEYS_LEN,
                });
            }
            Interaction::Hovered => {
                *color = HOVERED_BUTTON.into();
                border_color.0 = Color::WHITE;
            }
            Interaction::None => {
                *color = NORMAL_BUTTON.into();
                border_color.0 = Color::BLACK;
            }
        }
    }
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
    if sort_command.requested {}

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
