use crate::{
    attachment::{Attachment, GraphicalAttachment},
    data::internal::DataUniformBuilder,
    point_cloud::{PCSettings, geometry::PointCloudDesc},
    shape::{DataMut, FixedRenderer, ShapeGeometry, ShapeSettings},
    window::InnerGraphicalState,
};
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};

pub type Points<S> = Attachment<S, PointCloudDesc>;

impl GraphicalAttachment for Points<InnerGraphicalState> {
    fn move_elements(&mut self, queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]) {
        for (buffer_index, self_index) in self.position_indices.iter().enumerate() {
            if let Some(index) = indices.iter().position(|p| p == self_index) {
                self.geometry.positions[buffer_index] = pos[index];
                self.renderer
                    .fixed
                    .update_vertex(queue, buffer_index as u32, &self.geometry);
            }
        }
    }

    fn render<'c, 'd>(&'c self, render_pass: &mut wgpu::RenderPass<'d>)
    where
        'c: 'd,
    {
        if self.geometry.get_total_elements() > 0 {
            self.renderer.render_attached(render_pass);
        }
    }

    fn draw_ui(
        &mut self,
        ui: &mut egui::Ui,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _camera_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
        refresh_screen: &mut bool,
    ) {
        if self.settings.draw_ui(ui, &mut false) {
            *refresh_screen = true;
            self.settings
                .refresh_buffer(queue, &self.renderer.settings_uniform);
        }
    }
}

pub type PointsSettingsMut<'a, Ctxt> = DataMut<'a, &'a mut PCSettings, Ctxt>;
