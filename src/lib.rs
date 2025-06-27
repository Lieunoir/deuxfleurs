#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]
mod aabb;
pub mod attachment;
mod camera;
pub mod data;
mod deferred;
mod geometry;
mod obj_load;
mod picker;
pub mod point_cloud;
mod resources;
mod screenshot;
pub mod segment;
mod settings;
mod shader;
pub mod surface;
mod texture;
/// General types for genericity in functions parameters.
pub mod types;
///  Custom Ui components for mesh loading
pub mod ui;
mod util;
mod window;
use crate::point_cloud::DisplayPointCloud;
use crate::segment::DisplaySegment;
use crate::surface::geometry::DisplaySurface;
pub use egui;
use indexmap::IndexMap;
pub use resources::{load_mesh, load_mesh_blocking};
pub use settings::Settings;
pub use wgpu::Color;
pub use window::{InitialState, RunningState};
use window::{InnerBareState, State};

/// Re exported types for visibility
pub mod internal {
    pub use crate::geometry::{Element, ElementMut};
    pub use crate::window::State;
}

/// First initialization of the app. The resulting [`InitialState`]
/// can then be used to register geometries and data. It the has to
/// be ran.
///
/// Arguments:
/// * `settings`: global app [`Settings`]
#[must_use]
pub fn init(settings: Settings) -> InitialState {
    State::new_inner(InnerBareState {
        surfaces: IndexMap::new(),
        clouds: IndexMap::new(),
        segments: IndexMap::new(),
        settings,
    })
}
