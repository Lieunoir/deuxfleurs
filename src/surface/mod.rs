mod attachment;
mod data;
pub(crate) mod geometry;
mod shader;

pub(crate) use attachment::{NewSurfaceAttachment, SurfaceAttachment};
pub use data::VertexScalarSettingsMut;
pub(crate) use geometry::{DisplaySurface, UninitedSurface};
pub use geometry::{Surface, SurfaceMut};
