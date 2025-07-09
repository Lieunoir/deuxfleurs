#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};

use crate::{
    Settings,
    attachment::{
        NewPoints, NewSegments, NewVectorField, Points, Segments, VectorField, VectorFieldSettings,
        internal::AttachmentPosition,
    },
    point_cloud::PCSettings,
    segment::PCSettings as SegmentSettings,
    shape::{AttachedGeometry, GraphicalContext, NewAttachedGeometry},
};

#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub enum NewSurfaceAttachment {
    VectorField(NewVectorField),
    Points(NewPoints),
    Segments(NewSegments),
}

pub enum SurfaceAttachmentSettings<'a> {
    VectorField(&'a mut VectorFieldSettings),
    Points(&'a mut PCSettings),
    Segments(&'a mut SegmentSettings),
}

pub enum SurfaceAttachmentArgs {
    VectorField((Vec<[f32; 3]>, Vec<[f32; 3]>)),
    Points((Vec<u32>, Vec<[f32; 3]>)),
    Segments((Vec<u32>, (Vec<[f32; 3]>, Vec<[u32; 2]>))),
}

pub enum SurfaceAttachment {
    VectorField(VectorField),
    Points(Points),
    Segments(Segments),
}

impl<'a> AttachedGeometry<&'a mut Settings> for NewSurfaceAttachment {
    type Args = SurfaceAttachmentArgs;
    type Settings<'b> = SurfaceAttachmentSettings<'b>;

    fn new(
        name: String,
        args: Self::Args,
        position: AttachmentPosition,
        characteristic_l: f32,
        context: &mut &'a mut Settings,
        transform_layout: &(),
    ) -> Self {
        match args {
            SurfaceAttachmentArgs::VectorField(args) => {
                NewSurfaceAttachment::VectorField(<NewVectorField as AttachedGeometry<
                    &mut Settings,
                >>::new(
                    name,
                    args,
                    position,
                    characteristic_l,
                    context,
                    transform_layout,
                ))
            }
            SurfaceAttachmentArgs::Points(args) => NewSurfaceAttachment::Points(NewPoints::new(
                name,
                args,
                position,
                characteristic_l,
                context,
                transform_layout,
            )),
            SurfaceAttachmentArgs::Segments(args) => {
                NewSurfaceAttachment::Segments(NewSegments::new(
                    name,
                    args,
                    position,
                    characteristic_l,
                    context,
                    transform_layout,
                ))
            }
        }
    }

    fn show(&mut self, show: bool, refresh_screen: &mut bool) {
        match self {
            NewSurfaceAttachment::VectorField(v) => v.show(show, refresh_screen),
            NewSurfaceAttachment::Points(p) => p.show(show, refresh_screen),
            NewSurfaceAttachment::Segments(s) => s.show(show, refresh_screen),
        }
    }

    fn shown(&self) -> bool {
        match self {
            NewSurfaceAttachment::VectorField(v) => v.shown(),
            NewSurfaceAttachment::Points(p) => p.shown(),
            NewSurfaceAttachment::Segments(s) => s.shown(),
        }
    }

    fn get_settings(&mut self) -> Self::Settings<'_> {
        match self {
            NewSurfaceAttachment::VectorField(v) => {
                SurfaceAttachmentSettings::VectorField(v.get_settings())
            }
            NewSurfaceAttachment::Points(p) => SurfaceAttachmentSettings::Points(p.get_settings()),
            NewSurfaceAttachment::Segments(s) => {
                SurfaceAttachmentSettings::Segments(s.get_settings())
            }
        }
    }

    fn get_attached_position(&self) -> &AttachmentPosition {
        match self {
            NewSurfaceAttachment::VectorField(v) => v.get_attached_position(),
            NewSurfaceAttachment::Points(p) => p.get_attached_position(),
            NewSurfaceAttachment::Segments(s) => s.get_attached_position(),
        }
    }

    fn move_elements(&mut self, queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]) {
        match self {
            NewSurfaceAttachment::VectorField(v) => v.move_elements(queue, indices, pos),
            NewSurfaceAttachment::Points(p) => p.move_elements(queue, indices, pos),
            NewSurfaceAttachment::Segments(s) => s.move_elements(queue, indices, pos),
        }
    }
}

impl NewAttachedGeometry for NewSurfaceAttachment {
    type UpgradedAttachedGeometry = SurfaceAttachment;

    fn init(
        self,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self::UpgradedAttachedGeometry {
        match self {
            NewSurfaceAttachment::VectorField(v) => SurfaceAttachment::VectorField(v.init(
                device,
                camera_light_bind_group_layout,
                transform_bind_group_layout,
                color_format,
            )),
            NewSurfaceAttachment::Points(p) => SurfaceAttachment::Points(p.init(
                device,
                camera_light_bind_group_layout,
                transform_bind_group_layout,
                color_format,
            )),
            NewSurfaceAttachment::Segments(p) => SurfaceAttachment::Segments(p.init(
                device,
                camera_light_bind_group_layout,
                transform_bind_group_layout,
                color_format,
            )),
        }
    }

    fn downgrade(upgraded: &Self::UpgradedAttachedGeometry) -> Self {
        match upgraded {
            SurfaceAttachment::VectorField(v) => {
                NewSurfaceAttachment::VectorField(NewVectorField::downgrade(v))
            }
            SurfaceAttachment::Points(p) => NewSurfaceAttachment::Points(NewPoints::downgrade(p)),
            SurfaceAttachment::Segments(s) => {
                NewSurfaceAttachment::Segments(NewSegments::downgrade(s))
            }
        }
    }
}

impl<'a> AttachedGeometry<GraphicalContext<'a>> for SurfaceAttachment {
    type Args = SurfaceAttachmentArgs;
    type Settings<'b> = SurfaceAttachmentSettings<'b>;

    fn new(
        name: String,
        args: Self::Args,
        position: AttachmentPosition,
        characteristic_l: f32,
        context: &mut GraphicalContext<'a>,
        transform_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        match args {
            SurfaceAttachmentArgs::VectorField(args) => {
                SurfaceAttachment::VectorField(<VectorField as AttachedGeometry<
                    GraphicalContext<'a>,
                >>::new(
                    name,
                    args,
                    position,
                    characteristic_l,
                    context,
                    transform_layout,
                ))
            }
            SurfaceAttachmentArgs::Points(args) => SurfaceAttachment::Points(Points::new(
                name,
                args,
                position,
                characteristic_l,
                context,
                transform_layout,
            )),
            SurfaceAttachmentArgs::Segments(args) => SurfaceAttachment::Segments(Segments::new(
                name,
                args,
                position,
                characteristic_l,
                context,
                transform_layout,
            )),
        }
    }

    fn show(&mut self, show: bool, refresh_screen: &mut bool) {
        match self {
            SurfaceAttachment::VectorField(v) => v.show(show, refresh_screen),
            SurfaceAttachment::Points(p) => p.show(show, refresh_screen),
            SurfaceAttachment::Segments(s) => s.show(show, refresh_screen),
        }
    }

    fn shown(&self) -> bool {
        match self {
            SurfaceAttachment::VectorField(v) => v.shown(),
            SurfaceAttachment::Points(p) => p.shown(),
            SurfaceAttachment::Segments(s) => s.shown(),
        }
    }

    fn get_settings(&mut self) -> Self::Settings<'_> {
        match self {
            SurfaceAttachment::VectorField(v) => {
                SurfaceAttachmentSettings::VectorField(v.get_settings())
            }
            SurfaceAttachment::Points(p) => SurfaceAttachmentSettings::Points(p.get_settings()),
            SurfaceAttachment::Segments(s) => SurfaceAttachmentSettings::Segments(s.get_settings()),
        }
    }

    fn get_attached_position(&self) -> &AttachmentPosition {
        match self {
            SurfaceAttachment::VectorField(v) => v.get_attached_position(),
            SurfaceAttachment::Points(p) => p.get_attached_position(),
            SurfaceAttachment::Segments(s) => s.get_attached_position(),
        }
    }

    fn move_elements(&mut self, queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]) {
        match self {
            SurfaceAttachment::VectorField(v) => v.move_elements(queue, indices, pos),
            SurfaceAttachment::Points(p) => p.move_elements(queue, indices, pos),
            SurfaceAttachment::Segments(s) => s.move_elements(queue, indices, pos),
        }
    }

    fn render<'c, 'd>(&'c self, render_pass: &mut wgpu::RenderPass<'d>)
    where
        'c: 'd,
    {
        match self {
            SurfaceAttachment::VectorField(v) => v.render(render_pass),
            SurfaceAttachment::Points(p) => p.render(render_pass),
            SurfaceAttachment::Segments(s) => s.render(render_pass),
        }
    }

    fn draw_ui(
        &mut self,
        ui: &mut egui::Ui,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        refresh_screen: &mut bool,
    ) {
        match self {
            SurfaceAttachment::VectorField(v) => v.draw_ui(
                ui,
                device,
                queue,
                camera_light_bind_group_layout,
                color_format,
                refresh_screen,
            ),
            SurfaceAttachment::Points(p) => p.draw_ui(
                ui,
                device,
                queue,
                camera_light_bind_group_layout,
                color_format,
                refresh_screen,
            ),
            SurfaceAttachment::Segments(s) => s.draw_ui(
                ui,
                device,
                queue,
                camera_light_bind_group_layout,
                color_format,
                refresh_screen,
            ),
        }
    }
}
