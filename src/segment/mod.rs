pub(crate) mod geometry;
mod shader;
mod sphere_shader;

pub(crate) use geometry::{DisplaySegment, PCSettings, UninitedSegment};
pub use geometry::{Segment, SegmentMut};
