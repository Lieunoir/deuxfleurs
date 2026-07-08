use crate::{
    attachment::internal::AttachmentPosition, data::internal::DataUniform, window::ContextHolder,
};

use super::DataUniformBuilder;

pub struct DataMut<'a, T, S: ContextHolder> {
    pub(crate) inner: T,
    pub(crate) context: S::Context<'a>,
    pub(crate) uniform: &'a S::DataUniform,
}

impl<'a, T, S: ContextHolder> DataMut<'a, T, S> {
    pub(crate) fn convert<U, F: FnOnce(T) -> U>(self, f: F) -> DataMut<'a, U, S> {
        DataMut {
            inner: f(self.inner),
            uniform: self.uniform,
            context: self.context,
        }
    }
}

impl<'a, T: DataUniformBuilder, S: ContextHolder> DataMut<'a, &'a mut T, S> {
    pub(crate) fn update_data_settings(&mut self) {
        S::rebuild_data_uniform(self.inner, self.uniform, &self.context);
    }
}

pub trait GraphicalAttachedGeometry {
    type Downgraded;

    fn init(
        this: Self::Downgraded,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        transform_uniform: &DataUniform,
    ) -> Self;

    fn downgrade(&self) -> Self::Downgraded;

    fn draw_ui(&mut self, ui: &mut egui::Ui, queue: &wgpu::Queue, refresh_screen: &mut bool);

    fn move_elements(&mut self, queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]);

    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b;
}

impl<S: ContextHolder> AttachedGeometry<S> for () {
    type Args = ();

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

    fn get_attached_position(&self) -> &AttachmentPosition {
        &AttachmentPosition::Vertex
    }
}

impl GraphicalAttachedGeometry for () {
    type Downgraded = ();

    fn init(
        _this: (),
        _device: &wgpu::Device,
        _camera_bind_group_layout: &wgpu::BindGroupLayout,
        _transform_uniform: &DataUniform,
    ) -> Self {
        ()
    }

    fn downgrade(&self) -> Self::Downgraded {
        ()
    }

    fn draw_ui(&mut self, _ui: &mut egui::Ui, _queue: &wgpu::Queue, _refresh_screen: &mut bool) {}

    fn move_elements(&mut self, _queue: &wgpu::Queue, _indices: &[u32], _pos: &[[f32; 3]]) {}

    fn render<'a, 'b>(&'a self, _render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
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

    fn get_attached_position(&self) -> &AttachmentPosition;
}
