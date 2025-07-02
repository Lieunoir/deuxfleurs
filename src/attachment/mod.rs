mod vector_field;
mod vector_shader;

pub use vector_field::VectorFieldSettingsMut;
pub(crate) use vector_field::{NewVectorField, VectorField, VectorFieldSettings};

pub(crate) mod internal {
    pub enum AttachmentPosition {
        Vertex,
        Edge,
        Face,
    }
}
