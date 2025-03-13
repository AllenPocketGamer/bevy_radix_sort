use bevy::prelude::*;
use bevy_radix_sort::{GetSubgroupSizePlugin, SubgroupSize};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GetSubgroupSizePlugin)
        .add_systems(Startup, show_subgroup_size)
        .run();
}

fn show_subgroup_size(mut commands: Commands, subgroup_size: Res<SubgroupSize>) {
    commands.spawn(Camera2d::default());

    commands
        .spawn((Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },))
        .with_children(|parent| {
            parent.spawn((
                Text::new(format!("subgroup_size: {}", subgroup_size.0)),
                TextFont {
                    font_size: 30.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
        });
}
