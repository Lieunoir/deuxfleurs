mod points;
mod segments;
mod vector_field;
mod vector_shader;

pub use points::PointsSettingsMut;
pub(crate) use points::{NewPoints, Points};
pub use segments::SegmentsSettingsMut;
pub(crate) use segments::{NewSegments, Segments};
pub use vector_field::VectorFieldSettingsMut;
pub(crate) use vector_field::{NewVectorField, VectorField, VectorFieldSettings};

pub(crate) mod internal {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Serialize, Deserialize)]
    pub enum AttachmentPosition {
        Vertex,
        Edge,
        Face,
    }
}
