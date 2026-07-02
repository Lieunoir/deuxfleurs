mod points;
mod segments;
mod vector_field;
use crate::{
    attachment::internal::AttachmentPosition,
    data::internal::DataUniform,
    shape::{
        AttachedGeometry, AttachedRenderer, NewAttachedGeometry, ShapeDescriptor, ShapeGeometry,
        data::ShapeSettings,
    },
    window::{ContextHolder, InnerBareState, InnerGraphicalState},
};
pub(crate) use points::Points;
pub use points::PointsSettingsMut;
pub(crate) use segments::Segments;
pub use segments::SegmentsSettingsMut;
pub use vector_field::VectorFieldSettingsMut;
pub(crate) use vector_field::{VectorField, VectorFieldSettings};

pub(crate) mod internal {
    #[cfg(feature = "saves")]
    use serde::{Deserialize, Serialize};

    #[derive(Clone)]
    #[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
    pub enum AttachmentPosition {
        Vertex,
        Edge,
        Face,
    }
}

pub struct Attachment<S: ContextHolder + ?Sized, Desc: ShapeDescriptor> {
    show: bool,
    settings: Desc::Settings,
    position: AttachmentPosition,
    position_indices: Vec<u32>, //, included in trait
    geometry: Desc::Geometry,
    renderer: S::AttachedRenderer<Desc>,
}

impl<Desc: ShapeDescriptor> Clone for Attachment<InnerBareState, Desc> {
    fn clone(&self) -> Self {
        Self {
            show: self.show,
            settings: self.settings.clone(),
            position: self.position.clone(),
            position_indices: self.position_indices.clone(),
            geometry: self.geometry.clone(),
            renderer: self.renderer.clone(),
        }
    }
}

pub trait GraphicalAttachment: AttachedGeometry<InnerGraphicalState> {
    fn move_elements(&mut self, queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]);

    fn render<'c, 'd>(&'c self, render_pass: &mut wgpu::RenderPass<'d>)
    where
        'c: 'd;

    fn draw_ui(
        &mut self,
        ui: &mut egui::Ui,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        refresh_screen: &mut bool,
    );
}

impl<S: ContextHolder, Desc: ShapeDescriptor> crate::shape::AttachedGeometry<S>
    for Attachment<S, Desc>
{
    type Args = (Vec<u32>, <Desc::Geometry as ShapeGeometry>::Args);
    type Settings<'b>
        = &'b mut Desc::Settings
    where
        S: 'b,
        Desc: 'b,
        Desc::Settings: 'b;

    fn new(
        name: String,
        args: Self::Args,
        position: AttachmentPosition,
        characteristic_l: f32,
        context: &mut S::Context<'_>,
        transform_uniform: &S::TransformUniform,
    ) -> Self {
        S::notify_refresh_screen(context, true);
        let geometry = Desc::Geometry::new(args.1);
        let settings = Desc::Settings::new(&name, characteristic_l);
        let renderer = S::build_attached_renderer(&geometry, &settings, transform_uniform, context);
        Attachment {
            show: true,
            settings,
            geometry,
            renderer,
            position_indices: args.0,
            position,
        }
    }

    fn show(&mut self, show: bool, refresh_screen: &mut bool) {
        if self.show != show {
            *refresh_screen = true;
        }
        self.show = show;
    }

    fn shown(&self) -> bool {
        self.show
    }

    fn get_attached_position(&self) -> &AttachmentPosition {
        &self.position
    }

    fn get_settings(&mut self) -> Self::Settings<'_> {
        &mut self.settings
    }
}

impl<Desc: ShapeDescriptor> NewAttachedGeometry for Attachment<InnerBareState, Desc> {
    type UpgradedAttachedGeometry = Attachment<InnerGraphicalState, Desc>;

    fn init(
        self,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        transform_uniform: &DataUniform,
    ) -> Self::UpgradedAttachedGeometry {
        let renderer = AttachedRenderer::new(
            device,
            &self.geometry,
            &self.settings,
            transform_uniform,
            camera_bind_group_layout,
            &transform_uniform.bind_group_layout, // White lie
        );

        Self::UpgradedAttachedGeometry {
            renderer,
            show: self.show,
            geometry: self.geometry,
            settings: self.settings,
            position: self.position,
            position_indices: self.position_indices,
        }
    }

    fn downgrade(upgraded: &Self::UpgradedAttachedGeometry) -> Self {
        Self {
            renderer: (),
            show: upgraded.show,
            geometry: upgraded.geometry.clone(),
            settings: upgraded.settings.clone(),
            position: upgraded.position.clone(),
            position_indices: upgraded.position_indices.clone(),
        }
    }
}
