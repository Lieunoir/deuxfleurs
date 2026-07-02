use crate::{
    attachment::{
        Points, Segments, VectorField, VectorFieldSettings, internal::AttachmentPosition,
    },
    data::internal::DataUniform,
    point_cloud::PCSettings,
    segment::PCSettings as SegmentSettings,
    shape::{AttachedGeometry, GraphicalAttachedGeometry},
    window::{ContextHolder, InnerBareState, InnerGraphicalState},
};
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "saves",
    serde(bound = "VectorField<S>: Serialize + DeserializeOwned,
    Points<S>: Serialize + DeserializeOwned,
    Segments<S>: Serialize + DeserializeOwned,
")
)]
pub enum SurfaceAttachment<S: ContextHolder + ?Sized> {
    VectorField(VectorField<S>),
    Points(Points<S>),
    Segments(Segments<S>),
}

impl Clone for SurfaceAttachment<InnerBareState> {
    fn clone(&self) -> Self {
        match self {
            SurfaceAttachment::VectorField(v) => SurfaceAttachment::VectorField(v.clone()),
            SurfaceAttachment::Points(p) => SurfaceAttachment::Points(p.clone()),
            SurfaceAttachment::Segments(s) => SurfaceAttachment::Segments(s.clone()),
        }
    }
}

pub enum SurfaceAttachmentSettings<'a> {
    VectorField(&'a mut VectorFieldSettings),
    Points(&'a mut PCSettings),
    Segments(&'a mut SegmentSettings),
}

pub enum SurfaceAttachmentArgs {
    VectorField((Vec<u32>, (Vec<[f32; 3]>, Vec<[f32; 3]>))),
    Points((Vec<u32>, Vec<[f32; 3]>)),
    Segments((Vec<u32>, (Vec<[f32; 3]>, Vec<[u32; 2]>))),
}

impl<S: ContextHolder> AttachedGeometry<S> for SurfaceAttachment<S> {
    type Args = SurfaceAttachmentArgs;
    type Settings<'b>
        = SurfaceAttachmentSettings<'b>
    where
        S: 'b;

    fn new(
        name: String,
        args: Self::Args,
        position: AttachmentPosition,
        characteristic_l: f32,
        context: &mut S::Context<'_>,
        transform_uniform: &S::TransformUniform,
    ) -> Self {
        match args {
            SurfaceAttachmentArgs::VectorField(args) => {
                SurfaceAttachment::VectorField(VectorField::new(
                    name,
                    args,
                    position,
                    characteristic_l,
                    context,
                    transform_uniform,
                ))
            }
            SurfaceAttachmentArgs::Points(args) => SurfaceAttachment::Points(Points::new(
                name,
                args,
                position,
                characteristic_l,
                context,
                transform_uniform,
            )),
            SurfaceAttachmentArgs::Segments(args) => SurfaceAttachment::Segments(Segments::new(
                name,
                args,
                position,
                characteristic_l,
                context,
                transform_uniform,
            )),
        }
    }

    fn show(&mut self, show: bool, refresh_screen: &mut bool) {
        match self {
            SurfaceAttachment::VectorField(v) => {
                v.show(show, refresh_screen);
            }
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
}

impl GraphicalAttachedGeometry for SurfaceAttachment<InnerGraphicalState> {
    type Downgraded = SurfaceAttachment<InnerBareState>;

    fn init(
        this: Self::Downgraded,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        transform_uniform: &DataUniform,
    ) -> Self {
        match this {
            SurfaceAttachment::VectorField(v) => SurfaceAttachment::VectorField(VectorField::init(
                v,
                device,
                camera_bind_group_layout,
                transform_uniform,
            )),
            SurfaceAttachment::Points(p) => SurfaceAttachment::Points(Points::init(
                p,
                device,
                camera_bind_group_layout,
                transform_uniform,
            )),
            SurfaceAttachment::Segments(s) => SurfaceAttachment::Segments(Segments::init(
                s,
                device,
                camera_bind_group_layout,
                transform_uniform,
            )),
        }
    }

    fn downgrade(&self) -> Self::Downgraded {
        match self {
            SurfaceAttachment::VectorField(v) => SurfaceAttachment::VectorField(v.downgrade()),
            SurfaceAttachment::Points(p) => SurfaceAttachment::Points(p.downgrade()),
            SurfaceAttachment::Segments(s) => SurfaceAttachment::Segments(s.downgrade()),
        }
    }
    fn draw_ui(&mut self, ui: &mut egui::Ui, queue: &wgpu::Queue, refresh_screen: &mut bool) {
        match self {
            SurfaceAttachment::VectorField(v) => v.draw_ui(ui, queue, refresh_screen),
            SurfaceAttachment::Points(p) => p.draw_ui(ui, queue, refresh_screen),
            SurfaceAttachment::Segments(s) => s.draw_ui(ui, queue, refresh_screen),
        }
    }

    fn move_elements(&mut self, queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]) {
        match self {
            SurfaceAttachment::VectorField(v) => v.move_elements(queue, indices, pos),
            SurfaceAttachment::Points(p) => p.move_elements(queue, indices, pos),
            SurfaceAttachment::Segments(s) => s.move_elements(queue, indices, pos),
        }
    }

    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        match self {
            SurfaceAttachment::VectorField(v) => v.render(render_pass),
            SurfaceAttachment::Points(p) => p.render(render_pass),
            SurfaceAttachment::Segments(s) => s.render(render_pass),
        }
    }
}
