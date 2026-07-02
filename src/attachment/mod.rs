mod vector_field;
use crate::{
    attachment::internal::AttachmentPosition,
    data::internal::{DataUniform, DataUniformBuilder},
    point_cloud::geometry::PointCloudDesc,
    segment::geometry::SegmentDesc,
    shape::{
        AttachedGeometry, AttachedRenderer, DataMut, FixedRenderer, GraphicalAttachedGeometry,
        RenderAttached, ShapeDescriptor, ShapeGeometry, data::ShapeSettings,
    },
    window::{ContextHolder, InnerBareState, InnerGraphicalState},
};
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize, de::DeserializeOwned};
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

#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "saves",
    serde(bound = "Desc::Geometry: Serialize + DeserializeOwned,
        S::AttachedRenderer<Desc>: Serialize + DeserializeOwned,
        Desc::Settings: Serialize + DeserializeOwned,
        ")
)]
pub struct Attachment<S: ContextHolder + ?Sized, Desc: ShapeDescriptor> {
    show: bool,
    settings: Desc::Settings,
    position: AttachmentPosition,
    position_indices: Vec<u32>, //, included in trait
    geometry: Desc::Geometry,
    renderer: S::AttachedRenderer<Desc>,
}

pub type Points<S> = Attachment<S, PointCloudDesc>;
pub type PointsSettingsMut<'a, Ctxt> = DataMut<'a, &'a mut crate::point_cloud::PCSettings, Ctxt>;
pub type Segments<S> = Attachment<S, SegmentDesc>;
pub type SegmentsSettingsMut<'a, Ctxt> = DataMut<'a, &'a mut crate::segment::PCSettings, Ctxt>;

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

impl<S: ContextHolder, Desc: ShapeDescriptor> AttachedGeometry<S> for Attachment<S, Desc> {
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

impl<Desc: ShapeDescriptor> GraphicalAttachedGeometry for Attachment<InnerGraphicalState, Desc>
where
    AttachedRenderer<Desc>: RenderAttached,
{
    type Downgraded = Attachment<InnerBareState, Desc>;

    fn init(
        this: Self::Downgraded,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        transform_uniform: &DataUniform,
    ) -> Self {
        let renderer = AttachedRenderer::new(
            device,
            &this.geometry,
            &this.settings,
            transform_uniform,
            camera_bind_group_layout,
            &transform_uniform.bind_group_layout, // White lie
        );

        Self {
            renderer,
            show: this.show,
            geometry: this.geometry,
            settings: this.settings,
            position: this.position,
            position_indices: this.position_indices,
        }
    }

    fn downgrade(&self) -> Self::Downgraded {
        Attachment {
            renderer: (),
            show: self.show,
            geometry: self.geometry.clone(),
            settings: self.settings.clone(),
            position: self.position.clone(),
            position_indices: self.position_indices.clone(),
        }
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui, queue: &wgpu::Queue, refresh_screen: &mut bool) {
        if self.settings.draw_ui(ui, &mut false) {
            *refresh_screen = true;
            self.settings
                .refresh_buffer(queue, &self.renderer.settings_uniform);
        }
    }

    fn move_elements(&mut self, queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]) {
        for (buffer_index, self_index) in self.position_indices.iter().enumerate() {
            if let Some(index) = indices.iter().position(|p| p == self_index) {
                self.geometry.move_vertex(buffer_index as u32, pos[index]);
                self.renderer
                    .fixed
                    .update_vertex(queue, buffer_index as u32, &self.geometry);
            }
        }
    }

    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.geometry.get_total_elements() > 0 {
            self.renderer.render_attached(render_pass);
        }
    }
}
