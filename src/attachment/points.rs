#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};

use crate::{
    attachment::internal::AttachmentPosition,
    data::internal::DataUniformBuilder,
    point_cloud::{DisplayPointCloud, PCSettings, UninitedPointCloud},
    settings::Settings,
    shape::{
        AttachedGeometry, DataMut, FixedRenderer, GraphicalContext, NewAttachedGeometry,
        ShapeGeometry, ShapeSettings,
    },
};

#[derive(Clone)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct NewPoints {
    inner: UninitedPointCloud,
    indices: Vec<u32>,
    position: AttachmentPosition,
}

pub struct Points {
    inner: DisplayPointCloud,
    indices: Vec<u32>,
    position: AttachmentPosition,
}

impl<'a> AttachedGeometry<&'a mut Settings> for NewPoints {
    type Args = (Vec<u32>, Vec<[f32; 3]>);
    type Settings<'b> = &'b mut PCSettings;

    fn new(
        name: String,
        args: Self::Args,
        position: AttachmentPosition,
        characteristic_l: f32,
        _context: &mut &'a mut Settings,
        _transform_layout: &(),
    ) -> Self {
        let inner = UninitedPointCloud::new_bare(name, args.1, Some(3. * characteristic_l));
        NewPoints {
            inner,
            indices: args.0,
            position,
        }
    }

    fn get_settings(&mut self) -> Self::Settings<'_> {
        &mut self.inner.settings
    }

    fn show(&mut self, show: bool, _refresh_screen: &mut bool) {
        self.inner.show = show;
    }

    fn shown(&self) -> bool {
        self.inner.show
    }

    fn get_attached_position(&self) -> &AttachmentPosition {
        &self.position
    }

    fn move_elements(&mut self, _queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]) {
        for (buffer_index, self_index) in self.indices.iter().enumerate() {
            if let Some(index) = indices.iter().position(|p| p == self_index) {
                self.inner.geometry.positions[buffer_index] = pos[index];
            }
        }
    }
}

impl NewAttachedGeometry for NewPoints {
    type UpgradedAttachedGeometry = Points;

    fn init(
        self,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self::UpgradedAttachedGeometry {
        let inner = self.inner.upgrade(
            device,
            camera_light_bind_group_layout,
            transform_bind_group_layout, //Lie
            color_format,
        );
        Points {
            inner,
            indices: self.indices,
            position: self.position,
        }
    }

    fn downgrade(upgraded: &Self::UpgradedAttachedGeometry) -> Self {
        let inner = upgraded.inner.downgrade();
        NewPoints {
            inner,
            indices: upgraded.indices.clone(),
            position: upgraded.position.clone(),
        }
    }
}

impl<'a> AttachedGeometry<GraphicalContext<'a>> for Points {
    type Args = (Vec<u32>, Vec<[f32; 3]>);
    type Settings<'b> = &'b mut PCSettings;

    fn new(
        name: String,
        args: Self::Args,
        position: AttachmentPosition,
        characteristic_l: f32,
        context: &mut GraphicalContext<'a>,
        _transform_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        *context.refresh_screen = true;
        let inner = DisplayPointCloud::new(
            name,
            args.1,
            Some(3. * characteristic_l),
            context.device,
            context.camera_light_bind_group_layout,
            context.counter_bind_group_layout,
            context.color_format,
        );
        Points {
            inner,
            indices: args.0,
            position,
        }
    }

    fn show(&mut self, show: bool, refresh_screen: &mut bool) {
        if self.inner.show != show {
            *refresh_screen = true;
        }
        self.inner.show = show;
    }

    fn shown(&self) -> bool {
        self.inner.show
    }

    fn get_attached_position(&self) -> &AttachmentPosition {
        &self.position
    }

    fn get_settings(&mut self) -> Self::Settings<'_> {
        &mut self.inner.settings
    }

    fn move_elements(&mut self, queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]) {
        for (buffer_index, self_index) in self.indices.iter().enumerate() {
            if let Some(index) = indices.iter().position(|p| p == self_index) {
                self.inner.geometry.positions[buffer_index] = pos[index];
                self.inner.renderer.fixed.update_vertex(
                    queue,
                    buffer_index as u32,
                    &self.inner.geometry,
                );
            }
        }
    }

    fn render<'c, 'd>(&'c self, render_pass: &mut wgpu::RenderPass<'d>)
    where
        'c: 'd,
    {
        if self.inner.geometry().get_total_elements() > 0 {
            self.inner.renderer.render_attached(render_pass);
        }
    }

    fn draw_ui(
        &mut self,
        ui: &mut egui::Ui,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
        refresh_screen: &mut bool,
    ) {
        if self.inner.settings.draw_ui(ui, &mut false) {
            *refresh_screen = true;
            self.inner
                .settings
                .refresh_buffer(queue, &self.inner.renderer.settings_uniform);
        }
    }
}

pub type PointsSettingsMut<'a, Ctxt> = DataMut<'a, &'a mut PCSettings, Ctxt>;
