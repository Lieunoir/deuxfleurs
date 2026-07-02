use crate::{
    Settings,
    attachment::{GraphicalAttachment, internal::AttachmentPosition},
    data::internal::DataUniform,
    window::{ContextHolder, InnerBareState, InnerGraphicalState},
};

use super::DataUniformBuilder;

pub struct DataMut<'a, T, S: ContextHolder> {
    pub(crate) inner: T,
    pub(crate) context: S::Context<'a>,
    pub(crate) uniform: &'a S::DataUniform,
}

pub type UninitedData<'a, T> = DataMut<'a, T, InnerBareState>;
pub type DisplayData<'a, T> = DataMut<'a, T, InnerGraphicalState>;

impl<'a, T, S: ContextHolder> DataMut<'a, T, S> {
    pub(crate) fn convert<U, F: FnOnce(T) -> U>(self, f: F) -> DataMut<'a, U, S> {
        DataMut {
            inner: f(self.inner),
            uniform: self.uniform,
            context: self.context,
        }
    }
}

pub trait DataMutTrait {
    fn update_data_settings(&mut self);
}

impl<T, S> DataMutTrait for DataMut<'_, T, S>
where
    for<'a> S: ContextHolder<Context<'a> = &'a mut Settings>,
{
    fn update_data_settings(&mut self) {}
}

impl<T> DataMutTrait for DisplayData<'_, T>
where
    T: DataUniformBuilder,
{
    fn update_data_settings(&mut self) {
        self.uniform
            .as_ref()
            .map(|uniform| self.inner.refresh_buffer(self.context.queue, uniform));
    }
}

pub trait NewAttachedGeometry {
    type UpgradedAttachedGeometry;

    fn init(
        self,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        transform_uniform: &DataUniform,
    ) -> Self::UpgradedAttachedGeometry;

    fn downgrade(upgraded: &Self::UpgradedAttachedGeometry) -> Self;
}

impl<S: ContextHolder> AttachedGeometry<S> for () {
    type Args = ();
    type Settings<'a> = &'a mut ();

    fn new(
        _name: String,
        _args: Self::Args,
        _position: AttachmentPosition,
        _characteristic_l: f32,
        _context: &mut S::Context<'_>,
        _transform_layout: &S::TransformUniform,
    ) -> Self {
        ()
    }

    fn show(&mut self, _show: bool, _refresh_screen: &mut bool) {}

    fn shown(&self) -> bool {
        false
    }

    fn get_settings(&mut self) -> Self::Settings<'_> {
        self
    }

    fn get_attached_position(&self) -> &AttachmentPosition {
        &AttachmentPosition::Vertex
    }
}

impl NewAttachedGeometry for () {
    type UpgradedAttachedGeometry = ();

    fn init(
        self,
        _device: &wgpu::Device,
        _camera_bind_group_layout: &wgpu::BindGroupLayout,
        _transform_uniform: &DataUniform,
    ) -> Self::UpgradedAttachedGeometry {
        ()
    }

    fn downgrade(_upgraded: &Self::UpgradedAttachedGeometry) -> Self {
        ()
    }
}

impl GraphicalAttachment for () {
    fn draw_ui(
        &mut self,
        _ui: &mut egui::Ui,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _camera_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
        _refresh_screen: &mut bool,
    ) {
    }
    fn move_elements(&mut self, _queue: &wgpu::Queue, _indices: &[u32], _pos: &[[f32; 3]]) {}
    fn render<'c, 'd>(&'c self, _render_pass: &mut wgpu::RenderPass<'d>)
    where
        'c: 'd,
    {
    }
}

pub trait ShapeSettings: DataUniformBuilder + Clone {
    fn new(name: &str, characteristic_length: f32) -> Self;

    fn draw_ui(&mut self, ui: &mut egui::Ui, rebuild_pipeline: &mut bool) -> bool;
}

pub trait ShapeGeometry: Clone {
    type Args;

    fn new(args: Self::Args) -> Self;

    fn can_be_replaced_by(&self, _other: &Self) -> bool;

    fn get_positions(&self) -> &[[f32; 3]];

    fn get_total_elements(&self) -> u32;

    fn get_vertex_pos(&self, vertex: u32) -> [f32; 3];

    fn move_vertex(
        &mut self,
        vertex: u32,
        pos: [f32; 3],
    ) -> ((Vec<u32>, Vec<[f32; 3]>), (Vec<u32>, Vec<[f32; 3]>));

    fn get_characteristic_length(&self) -> f32;
}

pub trait AttachedGeometry<S: ContextHolder + ?Sized> {
    type Args;
    type Settings<'a>
    where
        Self: 'a;

    fn new(
        name: String,
        args: Self::Args,
        _position: AttachmentPosition,
        characteristic_l: f32,
        context: &mut S::Context<'_>,
        transform_layout: &S::TransformUniform,
    ) -> Self;

    fn shown(&self) -> bool;

    fn show(&mut self, _show: bool, _refresh_screen: &mut bool);

    fn get_settings(&mut self) -> Self::Settings<'_>;

    fn get_attached_position(&self) -> &AttachmentPosition;
}
