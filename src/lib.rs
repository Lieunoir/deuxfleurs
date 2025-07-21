#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]
//! # Controls
//! Currently touch constrols are only barely working.
//!
//! * Left click: rotate shape
//! * Right click: pan shape
//! * Mouse wheel: zoom/dezoom
//! * `Ctrl`+`+`: zoom UI
//! * `Ctrl`+`-`: dezoom UI
//! * `Ctrl`+`C`: save current camera state to clipboard
//! * `Ctrl`+`V`: load camera state from clipboard

/// Data associated to a shape which have their own renderer.
/// Wether they are shown or not does not affect other data.
pub mod attachment;
mod camera;
/// Data rendered directly onto the associated shape. Only
/// one can be displayed at a time on the corresponding
/// shape.
pub mod data;
mod obj_load;
/// Picked element types
pub mod picker;
/// Point clouds structs and associated data/settings
pub mod point_cloud;
mod post_process;
mod resources;
mod sbv;
mod screenshot;
/// Segment lists structs and associated data/settings
pub mod segment;
mod settings;
mod shader;
mod shape;
/// Triangular surfaces structs and associated data/settings
pub mod surface;
mod texture;
/// General types for genericity in functions parameters.
pub mod types;
///  Custom Ui components for mesh loading
pub mod ui;
mod util;
mod window;
use crate::segment::DisplaySegment;
use crate::surface::geometry::DisplaySurface;
use crate::{camera::Camera, point_cloud::DisplayPointCloud};
pub use egui;
use indexmap::IndexMap;
pub use resources::{load_mesh, load_mesh_blocking};
pub use settings::Settings;
pub use wgpu::Color;
pub use window::{InitialState, RunningState};
use window::{InnerBareState, State};

/// Re exported types for visibility
pub mod internal {
    pub use crate::shape::{Shape, ShapeMut};
    pub use crate::window::State;
}

/// First initialization of the app. The resulting [`InitialState`]
/// can then be used to register geometries and data. It then has
/// to be ran to be displayed.
///
/// Settings can also be parameterized by modifying [`get_settings_mut`]
#[must_use]
pub fn init() -> InitialState<impl FnMut(&mut egui::Ui, &mut RunningState)> {
    State::new_inner(InnerBareState {
        surfaces: IndexMap::new(),
        clouds: IndexMap::new(),
        segments: IndexMap::new(),
        settings: Settings::default(),
        callback: |_, _| {},
        camera: Camera::new(1.),
    })
}
