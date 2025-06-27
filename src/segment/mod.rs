pub(crate) mod geometry;
mod shader;
mod sphere_shader;

pub(crate) use geometry::{
    DisplaySegment, SegmentDataBuffer, SegmentFixedRenderer, SegmentPipeline, UninitedSegment,
};
pub use geometry::{Segment, SegmentMut};
