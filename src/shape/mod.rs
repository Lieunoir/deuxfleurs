use crate::attachment::internal::AttachmentPosition;
use crate::camera::Camera;
use crate::data::TransformSettings;
use crate::data::internal::{DataSettings, DataUniformBuilder};
use crate::sbv::SBV;
use crate::window::{ContextHolder, ContextHolderTypes, InnerBareState, InnerGraphicalState};
use crate::{RunningState, Settings};
pub(crate) use data::*;
use indexmap::IndexMap;
pub(crate) use renderer::*;
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};
use std::ops::Deref;
mod data;
mod renderer;

pub struct GraphicalContext<'a> {
    pub(crate) settings: &'a Settings,
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) camera_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(crate) counter_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(crate) refresh_screen: &'a mut bool,
}

// `Renderer` can be `()` !
#[derive(Clone)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct Shape<Geometry, Renderer, Settings, Data, AttachedGeometry> {
    pub(crate) name: String,
    pub(crate) geometry: Geometry,
    pub(crate) renderer: Renderer,
    pub(crate) show: bool,
    pub(crate) transform: TransformSettings,
    pub(crate) settings: Settings,
    data: IndexMap<String, Data>,
    attached_data: IndexMap<String, AttachedGeometry>,
    shown_data: Option<String>,
    pub(crate) sbv: SBV,
    modification_stamp: u32,
}

/// Mainly accessors for publically read-only fields
impl<Geometry, Renderer, Settings, Data, AttachedGeometry>
    Shape<Geometry, Renderer, Settings, Data, AttachedGeometry>
{
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn shown(&self) -> bool {
        self.show
    }

    /// This stamp increases each time the geometry of the shape is modified.
    pub fn get_modification_stamp(&self) -> u32 {
        self.modification_stamp
    }

    pub fn get_data(&self, name: &str) -> Option<&Data> {
        self.data.get(name)
    }

    pub fn get_attached_shape(&self, name: &str) -> Option<&AttachedGeometry> {
        self.attached_data.get(name)
    }

    pub fn get_transform(&self) -> [[f32; 4]; 4] {
        self.transform.get_transform().to_cols_array_2d()
    }
}

pub type UninitedShape<Geometry, Settings, Data, AttachedGeometry> =
    Shape<Geometry, (), Settings, Data, AttachedGeometry>;

pub type DisplayShape<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedGeometry> =
    Shape<Geometry, Renderer<Fixed, DataB, Pipeline>, Settings, Data, AttachedGeometry>;

impl<Geometry, Settings, Data, Attached> UninitedShape<Geometry, Settings, Data, Attached>
where
    Geometry: ShapeGeometry,
    Settings: ShapeSettings,
    Attached: NewAttachedGeometry,
{
    pub(crate) fn new_bare(name: String, args: Geometry::Args, char_l: Option<f32>) -> Self {
        let geometry = Geometry::new(args);
        Self::new_bare_with_geometry(name, geometry, char_l)
    }

    pub(crate) fn new_bare_with_geometry(
        name: String,
        geometry: Geometry,
        char_l: Option<f32>,
    ) -> Self {
        let transform = TransformSettings::default();
        let char_l = char_l.unwrap_or_else(|| geometry.get_characteristic_length());
        let settings = Settings::new(&name, char_l);
        let sbv = SBV::new(geometry.get_positions());
        Self {
            geometry,
            renderer: (),
            transform,
            settings,
            data: IndexMap::new(),
            attached_data: IndexMap::new(),
            shown_data: None,
            name,
            show: true,
            sbv,
            modification_stamp: 0,
        }
    }

    pub(crate) fn upgrade<Fixed, DataB, Pipeline>(
        self,
        device: &wgpu::Device,
        camera: &Camera,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Shape<
        Geometry,
        Renderer<Fixed, DataB, Pipeline>,
        Settings,
        Data,
        Attached::UpgradedAttachedGeometry,
    >
    where
        Data: DataUniformBuilder,
        Fixed: FixedRenderer<Geometry = Geometry>,
        DataB: DataBuffer<Data = Data, Geometry = Geometry>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry>,
    {
        let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
        let renderer = Renderer::new(
            device,
            &self.geometry,
            &self.transform,
            &self.settings,
            data,
            camera_bind_group_layout,
            counter_bind_group_layout,
        );

        let attached_data = self
            .attached_data
            .into_iter()
            .map(|(name, field)| {
                (
                    name,
                    field.init(
                        device,
                        camera,
                        camera_bind_group_layout,
                        &renderer.transform_uniform.bind_group_layout,
                    ),
                )
            })
            .collect();

        Shape {
            name: self.name,
            geometry: self.geometry,
            show: self.show,
            transform: self.transform,
            settings: self.settings,
            data: self.data,
            attached_data,
            renderer,
            shown_data: self.shown_data,
            sbv: self.sbv,
            modification_stamp: self.modification_stamp,
        }
    }
}

impl<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>
    DisplayShape<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>
where
    Attached: AttachedGeometry<InnerGraphicalState>,
    Geometry: ShapeGeometry,
    Data: DataSettings,
    Settings: ShapeSettings,
    Fixed: FixedRenderer<Geometry = Geometry>,
    DataB: DataBuffer<Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry>,
    Renderer<Fixed, DataB, Pipeline>: Render,
{
    pub(crate) fn new(
        name: String,
        args: Geometry::Args,
        char_l: Option<f32>,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let geometry = Geometry::new(args);
        Self::new_with_geometry(
            name,
            geometry,
            char_l,
            device,
            camera_bind_group_layout,
            counter_bind_group_layout,
        )
    }

    pub(crate) fn new_with_geometry(
        name: String,
        geometry: Geometry,
        char_l: Option<f32>,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let transform = TransformSettings::default();
        let char_l = char_l.unwrap_or_else(|| geometry.get_characteristic_length());
        let settings = Settings::new(&name, char_l);
        let renderer = Renderer::new(
            device,
            &geometry,
            &transform,
            &settings,
            None,
            camera_bind_group_layout,
            counter_bind_group_layout,
        );
        let sbv = SBV::new(geometry.get_positions());
        Self {
            geometry,
            renderer,
            transform,
            settings,
            data: IndexMap::new(),
            attached_data: IndexMap::new(),
            shown_data: None,
            name,
            show: true,
            sbv,
            modification_stamp: 0,
        }
    }

    pub(crate) fn downgrade<OldAttachedGeometry>(
        &self,
    ) -> UninitedShape<Geometry, Settings, Data, OldAttachedGeometry>
    where
        OldAttachedGeometry: NewAttachedGeometry<UpgradedAttachedGeometry = Attached>,
    {
        let attached_data = self
            .attached_data
            .iter()
            .map(|(k, v)| (k.clone(), OldAttachedGeometry::downgrade(&v)))
            .collect();
        UninitedShape::<Geometry, Settings, Data, OldAttachedGeometry> {
            name: self.name.clone(),
            geometry: self.geometry.clone(),
            data: self.data.clone(),
            show: self.show,
            sbv: self.sbv.clone(),
            transform: self.transform.clone(),
            settings: self.settings.clone(),
            shown_data: self.shown_data.clone(),
            renderer: (),
            attached_data,
            modification_stamp: self.modification_stamp,
        }
    }

    pub(crate) fn draw_ui(
        &mut self,
        ui: &mut egui::Ui,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        refresh_screen: &mut bool,
    ) {
        // While it may look like some graphical operations could be batched together,
        // since there is usually one ui interaction at a time, this isn't needed
        if self
            .transform
            .draw_transform(ui, self.geometry.get_positions())
        {
            self.transform
                .to_raw()
                .refresh_buffer(queue, &self.renderer.transform_uniform);
            *refresh_screen = true;
        }
        let mut rebuild_pipeline = false;
        if self.settings.draw_ui(ui, &mut rebuild_pipeline) || rebuild_pipeline {
            self.settings
                .refresh_buffer(queue, &self.renderer.settings_uniform);
            if rebuild_pipeline {
                let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
                self.renderer.rebuild_pipeline(
                    device,
                    data,
                    &self.settings,
                    camera_bind_group_layout,
                );
            }
            *refresh_screen = true;
        }
        for (name, data) in &mut self.data {
            let active = self.shown_data.as_ref() == Some(&name);
            let id = ui.make_persistent_id(name);
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.selectable_label(active, name.clone()).clicked() {
                            let data = if !active {
                                self.shown_data = Some(name.clone());
                                Some(&*data)
                            } else {
                                self.shown_data = None;
                                None
                            };
                            self.renderer.build_data_buffer(
                                device,
                                &self.geometry,
                                data,
                                &self.settings,
                                camera_bind_group_layout,
                            );
                            *refresh_screen = true;
                        }
                    })
                })
                .body(|ui| {
                    // Triggered on ui settings change
                    if data.draw_ui(ui) && active {
                        if let Some(data_uniform) = self.renderer.get_data_uniform() {
                            data.refresh_buffer(queue, data_uniform);
                            *refresh_screen = true;
                        }
                    }
                });
        }

        for (name, field) in &mut self.attached_data {
            let id = ui.make_persistent_id(name);
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.checkbox(&mut field.shown(), name.clone()).changed() {
                            field.show(!field.shown(), refresh_screen);
                        }
                    });
                })
                .body(|ui| {
                    field.draw_ui(
                        ui,
                        device,
                        queue,
                        camera_bind_group_layout,
                        color_format,
                        refresh_screen,
                    );
                });
        }
    }

    pub(crate) fn draw_gizmo(
        &mut self,
        ui: &mut egui::Ui,
        view: glam::Mat4,
        proj: glam::Mat4,
        queue: &wgpu::Queue,
        gizmo_hovered: &mut bool,
        refresh_screen: &mut bool,
    ) {
        if self.transform.draw_gizmo(ui, view, proj, gizmo_hovered) {
            self.transform
                .to_raw()
                .refresh_buffer(queue, &self.renderer.transform_uniform);
            *refresh_screen = true;
        }
    }

    pub(crate) fn move_vertex(&mut self, queue: &wgpu::Queue, vertex: u32, pos: [f32; 3]) {
        self.modification_stamp += 1;
        let ((adj_faces, adj_faces_centers), (adj_edges, adj_edges_centers)) =
            self.geometry.move_vertex(vertex, pos);
        self.sbv.add_point(pos);
        self.renderer
            .fixed
            .update_vertex(queue, vertex, &self.geometry);
        for attachment in self.attached_data.values_mut() {
            match attachment.get_attached_position() {
                AttachmentPosition::Vertex => attachment.move_elements(queue, &[vertex], &[pos]),
                AttachmentPosition::Face => {
                    attachment.move_elements(queue, &adj_faces, &adj_faces_centers)
                }
                AttachmentPosition::Edge => {
                    attachment.move_elements(queue, &adj_edges, &adj_edges_centers)
                }
            }
        }
    }

    pub(crate) fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show && self.geometry().get_total_elements() > 0 {
            self.renderer.render(render_pass);
            for (_, attached) in &self.attached_data {
                if attached.shown() {
                    attached.render(render_pass);
                }
            }
        }
    }

    pub(crate) fn render_shadow<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show && self.geometry().get_total_elements() > 0 {
            self.renderer.render_shadow(render_pass);
        }
    }

    pub(crate) fn render_picker<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show && self.geometry().get_total_elements() > 0 {
            self.renderer.render_picker(render_pass);
        }
    }
}

pub trait ShapeTrait<S: ContextHolder> {
    type Data;
    type Geometry: ShapeGeometry;
    type Attached: AttachedGeometry<S>;

    fn replace(
        &mut self,
        args: <Self::Geometry as ShapeGeometry>::Args,
        context: &mut S::Context<'_>,
    ) -> bool;

    fn show(&mut self, show: bool, context: &mut S::Context<'_>);

    fn set_transform(&mut self, transform: [[f32; 4]; 4], context: &mut S::Context<'_>);

    fn set_data(&mut self, name: Option<String>, context: &mut S::Context<'_>);

    fn add_data<'a>(
        &'a mut self,
        name: String,
        data: Self::Data,
        context: S::Context<'a>,
    ) -> DataMut<'a, &'a mut Self::Data, S>;

    fn remove_data(&mut self, name: String, context: &mut S::Context<'_>);

    fn remove_attached_shape(&mut self, name: String, context: &mut S::Context<'_>);

    fn update_settings(&mut self, _context: &mut S::Context<'_>, _rebuild_pipeline: bool) {}

    fn update_transform(&mut self, _context: &mut S::Context<'_>) {}

    fn add_attached_geometry<'a>(
        &'a mut self,
        name: String,
        args: <Self::Attached as AttachedGeometry<S>>::Args,
        position: AttachmentPosition,
        context: S::Context<'a>,
    ) -> DataMut<'a, &'a mut Self::Attached, S>;
}

impl<Geometry, Settings, Data, AttachedG, U: FnMut(&mut egui::Ui, &mut RunningState)>
    ShapeTrait<InnerBareState<U>> for UninitedShape<Geometry, Settings, Data, AttachedG>
where
    Geometry: ShapeGeometry,
    AttachedG: AttachedGeometry<InnerBareState<U>>,
    Data: DataSettings,
    Settings: ShapeSettings,
    AttachedG: NewAttachedGeometry,
{
    type Geometry = Geometry;
    type Attached = AttachedG;
    type Data = Data;

    fn replace(
        &mut self,
        args: <Self::Geometry as ShapeGeometry>::Args,
        _context: &mut &mut crate::Settings,
    ) -> bool {
        let new_geometry = <Self::Geometry as ShapeGeometry>::new(args);
        if self.geometry().can_be_replaced_by(&new_geometry) {
            self.geometry = new_geometry;
            true
        } else {
            *self = Self::new_bare_with_geometry(self.name.clone(), new_geometry, None);
            false
        }
    }

    fn show(&mut self, show: bool, _context: &mut &mut crate::Settings) {
        self.show = show;
    }

    fn set_transform(&mut self, transform: [[f32; 4]; 4], _context: &mut &mut crate::Settings) {
        self.transform.set_transform(transform);
    }

    fn set_data(&mut self, name: Option<String>, _context: &mut &mut crate::Settings) {
        self.shown_data = name;
    }

    fn remove_data(&mut self, name: String, _context: &mut &mut crate::Settings) {
        self.data.shift_remove(&name);
        if self.shown_data == Some(name) {
            self.shown_data = None;
        }
    }

    fn remove_attached_shape(&mut self, name: String, _context: &mut &mut crate::Settings) {
        self.attached_data.shift_remove(&name);
    }

    fn add_data<'a>(
        &'a mut self,
        name: String,
        data: Self::Data,
        context: <InnerBareState<U> as ContextHolderTypes>::Context<'a>,
    ) -> DataMut<'a, &'a mut Self::Data, InnerBareState<U>> {
        let old_data = self.data.insert(name.clone(), data);
        let data = self.data.get_mut(&name).unwrap();
        old_data.map(|old| data.apply_previous_settings(old));
        DataMut {
            inner: data,
            uniform: (),
            context,
        }
    }

    fn add_attached_geometry<'a>(
        &'a mut self,
        name: String,
        args: <Self::Attached as AttachedGeometry<InnerBareState<U>>>::Args,
        position: AttachmentPosition,
        mut context: <InnerBareState<U> as ContextHolderTypes>::Context<'a>,
    ) -> DataMut<'a, &'a mut Self::Attached, InnerBareState<U>> {
        let geometry = AttachedG::new(
            name.clone(),
            args,
            position,
            self.geometry().get_characteristic_length(),
            &mut context,
            &(),
        );
        self.attached_data.insert(name.clone(), geometry);
        DataMut {
            inner: self.attached_data.get_mut(&name).unwrap(),
            uniform: (),
            context,
        }
    }
}

impl<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedG> ShapeTrait<InnerGraphicalState>
    for DisplayShape<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedG>
where
    AttachedG: AttachedGeometry<InnerGraphicalState>,
    Geometry: ShapeGeometry,
    Data: DataSettings,
    Settings: ShapeSettings,
    Fixed: FixedRenderer<Geometry = Geometry>,
    DataB: DataBuffer<Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry>,
    Renderer<Fixed, DataB, Pipeline>: Render,
{
    type Data = Data;
    type Geometry = Geometry;
    type Attached = AttachedG;

    fn replace(
        &mut self,
        args: <Self::Geometry as ShapeGeometry>::Args,
        context: &mut GraphicalContext<'_>,
    ) -> bool {
        let new_geometry = <Self::Geometry as ShapeGeometry>::new(args);
        if self.geometry().can_be_replaced_by(&new_geometry) {
            self.renderer.fixed = Fixed::initialize(context.device, &new_geometry);
            self.geometry = new_geometry;
            self.sbv = SBV::new(self.geometry.get_positions());
            true
        } else {
            *self = Self::new_with_geometry(
                self.name.clone(),
                new_geometry,
                None,
                context.device,
                context.camera_bind_group_layout,
                context.counter_bind_group_layout,
            );
            false
        }
    }

    fn show(&mut self, show: bool, context: &mut GraphicalContext<'_>) {
        if self.show != show {
            *context.refresh_screen = true;
            self.show = show;
        }
    }

    fn set_transform(&mut self, transform: [[f32; 4]; 4], context: &mut GraphicalContext<'_>) {
        self.transform.set_transform(transform);
        self.transform
            .to_raw()
            .refresh_buffer(context.queue, &self.renderer.transform_uniform);
    }

    fn remove_data(&mut self, name: String, context: &mut GraphicalContext<'_>) {
        self.data.shift_remove(&name);
        if self.shown_data == Some(name) {
            self.shown_data = None;
            *context.refresh_screen |= self.show;
        }
    }

    fn remove_attached_shape(&mut self, name: String, context: &mut GraphicalContext<'_>) {
        if let Some(data) = self.attached_data.shift_remove(&name) {
            *context.refresh_screen |= data.shown() && self.show
        }
    }

    fn set_data(&mut self, name: Option<String>, context: &mut GraphicalContext<'_>) {
        if self.shown_data != name {
            self.shown_data = name;
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            self.renderer.build_data_buffer(
                context.device,
                &self.geometry,
                data,
                &self.settings,
                context.camera_bind_group_layout,
            );
            *context.refresh_screen |= self.show;
        }
    }

    fn add_data<'a>(
        &'a mut self,
        name: String,
        data: Self::Data,
        context: <InnerGraphicalState as ContextHolderTypes>::Context<'a>,
    ) -> DataMut<'a, &'a mut Self::Data, InnerGraphicalState> {
        let old_data = self.data.insert(name.clone(), data);
        let data = self.data.get_mut(&name).unwrap();
        old_data.map(|old| data.apply_previous_settings(old));
        if self.shown_data.as_ref() == Some(&name) {
            self.renderer.build_data_buffer(
                context.device,
                &self.geometry,
                Some(data),
                &self.settings,
                context.camera_bind_group_layout,
            );
            *context.refresh_screen |= self.show;
        }
        DataMut {
            inner: data,
            context: context,
            uniform: &self.renderer.data_uniform,
        }
    }

    fn update_settings(&mut self, context: &mut GraphicalContext<'_>, rebuild_pipeline: bool) {
        self.settings
            .refresh_buffer(context.queue, &self.renderer.settings_uniform);
        if rebuild_pipeline {
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            self.renderer.rebuild_pipeline(
                context.device,
                data,
                &self.settings,
                context.camera_bind_group_layout,
            );
        }
    }

    fn add_attached_geometry<'a>(
        &'a mut self,
        name: String,
        args: <Self::Attached as AttachedGeometry<InnerGraphicalState>>::Args,
        position: AttachmentPosition,
        mut context: <InnerGraphicalState as ContextHolderTypes>::Context<'a>,
    ) -> DataMut<'a, &'a mut Self::Attached, InnerGraphicalState> {
        *context.refresh_screen = true;
        {
            let geometry = Self::Attached::new(
                name.clone(),
                args,
                position,
                self.geometry().get_characteristic_length(),
                &mut context,
                &self.renderer.transform_uniform.bind_group_layout,
            );
            self.attached_data.insert(name.clone(), geometry);
        }
        DataMut {
            inner: self.attached_data.get_mut(&name).unwrap(),
            context: context,
            uniform: &self.renderer.data_uniform,
        }
    }
}

pub struct ShapeMut<'a, Shape, S: ContextHolder> {
    pub(crate) inner: &'a mut Shape,
    pub(crate) context: S::Context<'a>,
}

impl<Shape: ShapeTrait<S>, S: ContextHolder> Deref for ShapeMut<'_, Shape, S> {
    type Target = Shape;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<Shape: ShapeTrait<S>, S: ContextHolder> ShapeMut<'_, Shape, S> {
    pub fn show(&mut self, show: bool) -> &mut Self {
        self.inner.show(show, &mut self.context);
        self
    }

    pub fn set_transform(&mut self, transform: [[f32; 4]; 4]) {
        self.inner.set_transform(transform, &mut self.context);
    }

    pub fn set_data<St: Into<String>>(&mut self, name: Option<St>) -> &mut Self {
        self.inner.set_data(name.map(Into::into), &mut self.context);
        self
    }

    pub fn remove_data<St: Into<String>>(&mut self, name: St) {
        self.inner.remove_data(name.into(), &mut self.context);
    }

    pub fn remove_attached_shape<St: Into<String>>(&mut self, name: St) {
        self.inner
            .remove_attached_shape(name.into(), &mut self.context);
    }

    pub(crate) fn add_data(
        &'_ mut self,
        name: String,
        data: Shape::Data,
    ) -> DataMut<'_, &'_ mut Shape::Data, S> {
        let ctxt = <S as ContextHolder>::reborrow_context(&mut self.context);
        self.inner.add_data(name, data, ctxt)
    }

    pub(crate) fn add_attached_geometry(
        &'_ mut self,
        name: String,
        args: <Shape::Attached as AttachedGeometry<S>>::Args,
        position: AttachmentPosition,
    ) -> DataMut<'_, &'_ mut Shape::Attached, S> {
        let ctxt = <S as ContextHolder>::reborrow_context(&mut self.context);
        self.inner.add_attached_geometry(name, args, position, ctxt)
    }

    pub(crate) fn update_settings(&mut self, rebuild_pipeline: bool) -> &mut Self {
        self.inner
            .update_settings(&mut self.context, rebuild_pipeline);
        self
    }
}
