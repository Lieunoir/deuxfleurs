use crate::Settings;
use crate::attachment::GraphicalAttachment;
use crate::attachment::internal::AttachmentPosition;
use crate::data::TransformSettings;
use crate::data::internal::{DataSettings, DataUniformBuilder};
use crate::sbv::SBV;
use crate::window::{ContextHolder, InnerBareState, InnerGraphicalState};
pub(crate) use data::*;
use indexmap::IndexMap;
pub(crate) use renderer::*;
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};
use std::ops::Deref;
pub(crate) mod data;
pub(crate) mod renderer;

pub struct GraphicalContext<'a> {
    pub(crate) settings: &'a Settings,
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) camera_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(crate) counter_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(crate) refresh_screen: &'a mut bool,
}

pub trait InvariantShapeDescriptor {
    type Geometry: ShapeGeometry;
    type Settings: ShapeSettings;
    type Data: DataSettings;
    type FixedBuffer: FixedRenderer<Geometry = Self::Geometry>;
    type DataBuffer: DataBuffer<Data = Self::Data, Geometry = Self::Geometry>;
    type Pipeline: RenderPipeline<Settings = Self::Settings, Data = Self::Data, Geometry = Self::Geometry>;
    type Attached<S: ContextHolder>: AttachedGeometry<S>;
}

#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct Shape<S: ContextHolder, Desc: InvariantShapeDescriptor + ?Sized> {
    pub(crate) name: String,
    pub(crate) geometry: Desc::Geometry,
    pub(crate) renderer: S::Renderer<Desc>,
    pub(crate) show: bool,
    pub(crate) transform: TransformSettings,
    pub(crate) settings: Desc::Settings,
    data: IndexMap<String, Desc::Data>,
    attached_data: IndexMap<String, Desc::Attached<S>>,
    shown_data: Option<String>,
    pub(crate) sbv: SBV,
    modification_stamp: u32,
}

impl<S: ContextHolder, Desc: InvariantShapeDescriptor> Clone for Shape<S, Desc>
where
    Desc::Attached<S>: Clone,
    S::Renderer<Desc>: Clone,
{
    fn clone(&self) -> Self {
        Shape {
            name: self.name.clone(),
            geometry: self.geometry.clone(),
            renderer: self.renderer.clone(),
            show: self.show,
            transform: self.transform.clone(),
            settings: self.settings.clone(),
            data: self.data.clone(),
            attached_data: self.attached_data.clone(),
            shown_data: self.shown_data.clone(),
            sbv: self.sbv.clone(),
            modification_stamp: self.modification_stamp,
        }
    }
}

/// Mainly accessors for publically read-only fields
impl<S: ContextHolder, Desc: InvariantShapeDescriptor> Shape<S, Desc> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn geometry(&self) -> &Desc::Geometry {
        &self.geometry
    }

    pub fn shown(&self) -> bool {
        self.show
    }

    /// This stamp increases each time the geometry of the shape is modified.
    pub fn get_modification_stamp(&self) -> u32 {
        self.modification_stamp
    }

    pub fn get_data(&self, name: &str) -> Option<&Desc::Data> {
        self.data.get(name)
    }

    pub fn get_attached_shape(&self, name: &str) -> Option<&Desc::Attached<S>> {
        self.attached_data.get(name)
    }

    pub fn get_transform(&self) -> [[f32; 4]; 4] {
        self.transform.get_transform().to_cols_array_2d()
    }
}

pub type UninitedShape<Desc> = Shape<InnerBareState, Desc>;
pub type DisplayShape<Desc> = Shape<InnerGraphicalState, Desc>;

impl<Desc: InvariantShapeDescriptor> UninitedShape<Desc>
where
    Desc::Attached<InnerBareState>: NewAttachedGeometry,
{
    pub(crate) fn upgrade(
        self,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> DisplayShape<Desc>
    where
        Desc: InvariantShapeDescriptor<
            Attached<InnerGraphicalState> = <<Desc as InvariantShapeDescriptor>::Attached<
                InnerBareState,
            > as NewAttachedGeometry>::UpgradedAttachedGeometry,
        >,
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
                        camera_bind_group_layout,
                        &renderer.transform_uniform,
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

impl<Desc> DisplayShape<Desc>
where
    Desc: InvariantShapeDescriptor,
    Renderer<Desc>: Render,
    Desc::Attached<InnerGraphicalState>: GraphicalAttachment,
{
    #[cfg(feature = "saves")]
    pub(crate) fn downgrade(&self) -> UninitedShape<Desc>
    where
        Desc: ShapeDescriptor<
                InnerBareState,
            >,
        <Desc as ShapeDescriptor<InnerBareState>>::AttachedGeometry:
            NewAttachedGeometry<UpgradedAttachedGeometry = <Desc as ShapeDescriptor<InnerGraphicalState>>::AttachedGeometry>,
    {
        let attached_data = self
            .attached_data
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    <Desc as ShapeDescriptor<InnerBareState>>::AttachedGeometry::downgrade(&v),
                )
            })
            .collect();
        UninitedShape::<Desc> {
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
        // since there is usually only one ui interaction at a time it is not needed
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
                            self.renderer.rebuild_data_buffer(
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

pub trait ShapeTrait<S: ContextHolder>: InvariantShapeDescriptor {
    fn new(
        name: String,
        args: <Self::Geometry as ShapeGeometry>::Args,
        char_l: Option<f32>,
        context: &mut S::Context<'_>,
    ) -> Shape<S, Self> {
        let geometry = Self::Geometry::new(args);
        Self::new_with_geometry(name, geometry, char_l, context)
    }

    fn new_with_geometry(
        name: String,
        geometry: Self::Geometry,
        char_l: Option<f32>,
        context: &mut S::Context<'_>,
    ) -> Shape<S, Self> {
        let transform = TransformSettings::default();
        let char_l = char_l.unwrap_or_else(|| geometry.get_characteristic_length());
        let settings = Self::Settings::new(&name, char_l);
        let renderer = S::build_renderer(&geometry, &transform, &settings, None, context);
        let sbv = SBV::new(geometry.get_positions());
        Shape {
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

    // Return value is true when replaced, false when created
    fn replace_or_create(
        this: &mut Shape<S, Self>,
        args: <Self::Geometry as ShapeGeometry>::Args,
        context: &mut S::Context<'_>,
    ) -> bool {
        let new_geometry = <Self::Geometry as ShapeGeometry>::new(args);
        if this.geometry.can_be_replaced_by(&new_geometry) {
            S::rebuild_fixed_buffer(&mut this.renderer, &new_geometry, context);
            this.geometry = new_geometry;
            this.sbv = SBV::new(this.geometry.get_positions());
            true
        } else {
            *this = Self::new_with_geometry(
                std::mem::take(&mut this.name),
                new_geometry,
                None,
                context,
            );
            false
        }
    }

    fn show(this: &mut Shape<S, Self>, show: bool, context: &mut S::Context<'_>) {
        if this.show != show {
            S::notify_refresh_screen(context, true);
            this.show = show;
        }
    }

    fn set_data(this: &mut Shape<S, Self>, name: Option<String>, context: &mut S::Context<'_>) {
        if this.shown_data != name {
            this.shown_data = name;
            let data = this.shown_data.as_ref().map(|d| this.data.get(d)).flatten();
            S::rebuild_data_buffer(
                &mut this.renderer,
                &this.geometry,
                data,
                &this.settings,
                context,
            );
            S::notify_refresh_screen(context, this.show);
        }
    }

    fn add_data<'a>(
        this: &'a mut Shape<S, Self>,
        name: String,
        data: Self::Data,
        mut context: S::Context<'a>,
    ) -> DataMut<'a, &'a mut Self::Data, S> {
        let old_data = this.data.insert(name.clone(), data);
        let data = this.data.get_mut(&name).unwrap();
        old_data.map(|old| data.apply_previous_settings(old));
        if this.shown_data.as_ref() == Some(&name) {
            S::rebuild_data_buffer(
                &mut this.renderer,
                &this.geometry,
                Some(data),
                &this.settings,
                &context,
            );
            S::notify_refresh_screen(&mut context, this.show);
        }
        DataMut {
            inner: data,
            context: context,
            uniform: S::get_renderer_data_uniform(&this.renderer),
        }
    }

    fn remove_data(this: &mut Shape<S, Self>, name: String, context: &mut S::Context<'_>) {
        if this.data.shift_remove(&name).is_some() && this.shown_data == Some(name) {
            this.shown_data = None;
            S::notify_refresh_screen(context, this.show);
        }
    }

    fn remove_attached_shape(
        this: &mut Shape<S, Self>,
        name: String,
        context: &mut S::Context<'_>,
    ) {
        if let Some(data) = this.attached_data.shift_remove(&name)
            && data.shown()
        {
            S::notify_refresh_screen(context, this.show);
        }
    }

    fn update_settings(
        this: &mut Shape<S, Self>,
        context: &mut S::Context<'_>,
        rebuild_pipeline: bool,
    ) {
        S::update_settings(&mut this.renderer, &this.settings, context);
        if rebuild_pipeline {
            let data = this.shown_data.as_ref().map(|d| this.data.get(d)).flatten();
            S::rebuild_pipeline(&mut this.renderer, data, &this.settings, context);
        }
    }

    fn set_transform(
        this: &mut Shape<S, Self>,
        transform: [[f32; 4]; 4],
        context: &mut S::Context<'_>,
    ) {
        this.transform.set_transform(transform);
        S::update_transform(&mut this.renderer, &this.transform, context);
    }

    fn add_attached_geometry<'a>(
        this: &'a mut Shape<S, Self>,
        name: String,
        args: <Self::Attached<S> as AttachedGeometry<S>>::Args,
        position: AttachmentPosition,
        context: S::Context<'a>,
    ) -> DataMut<'a, &'a mut Self::Attached<S>, S>;
}

impl<Desc> ShapeTrait<InnerBareState> for Desc
where
    Desc: InvariantShapeDescriptor,
    Desc::Attached<InnerBareState>: NewAttachedGeometry,
{
    fn add_attached_geometry<'a>(
        this: &'a mut UninitedShape<Desc>,
        name: String,
        args: <Desc::Attached<InnerBareState> as AttachedGeometry<InnerBareState>>::Args,
        position: AttachmentPosition,
        mut context: <InnerBareState as ContextHolder>::Context<'a>,
    ) -> DataMut<'a, &'a mut Desc::Attached<InnerBareState>, InnerBareState> {
        let geometry = Desc::Attached::new(
            name.clone(),
            args,
            position,
            this.geometry().get_characteristic_length(),
            &mut context,
            &(),
        );
        this.attached_data.insert(name.clone(), geometry);
        DataMut {
            inner: this.attached_data.get_mut(&name).unwrap(),
            uniform: (),
            context,
        }
    }
}

impl<Desc> ShapeTrait<InnerGraphicalState> for Desc
where
    Desc: InvariantShapeDescriptor,
{
    fn add_attached_geometry<'a>(
        this: &'a mut DisplayShape<Desc>,
        name: String,
        args: <Desc::Attached<InnerGraphicalState> as AttachedGeometry<InnerGraphicalState>>::Args,
        position: AttachmentPosition,
        mut context: <InnerGraphicalState as ContextHolder>::Context<'a>,
    ) -> DataMut<'a, &'a mut Desc::Attached<InnerGraphicalState>, InnerGraphicalState> {
        *context.refresh_screen = true;
        {
            let geometry = Desc::Attached::new(
                name.clone(),
                args,
                position,
                this.geometry.get_characteristic_length(),
                &mut context,
                &this.renderer.transform_uniform,
            );
            this.attached_data.insert(name.clone(), geometry);
        }
        DataMut {
            inner: this.attached_data.get_mut(&name).unwrap(),
            context: context,
            uniform: &this.renderer.data_uniform,
        }
    }
}

pub struct ShapeMut<'a, Shape, S: ContextHolder> {
    pub(crate) inner: &'a mut Shape,
    pub(crate) context: S::Context<'a>,
}

impl<Desc: ShapeTrait<S>, S: ContextHolder> Deref for ShapeMut<'_, Shape<S, Desc>, S> {
    type Target = Shape<S, Desc>;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<Desc: ShapeTrait<S>, S: ContextHolder> ShapeMut<'_, Shape<S, Desc>, S> {
    pub fn show(&mut self, show: bool) -> &mut Self {
        Desc::show(&mut self.inner, show, &mut self.context);
        self
    }

    pub fn set_transform(&mut self, transform: [[f32; 4]; 4]) {
        Desc::set_transform(&mut self.inner, transform, &mut self.context);
    }

    pub fn set_data<St: Into<String>>(&mut self, name: Option<St>) -> &mut Self {
        Desc::set_data(&mut self.inner, name.map(Into::into), &mut self.context);
        self
    }

    pub fn remove_data<St: Into<String>>(&mut self, name: St) {
        Desc::remove_data(&mut self.inner, name.into(), &mut self.context);
    }

    pub fn remove_attached_shape<St: Into<String>>(&mut self, name: St) {
        Desc::remove_attached_shape(&mut self.inner, name.into(), &mut self.context);
    }

    pub(crate) fn add_data(
        &'_ mut self,
        name: String,
        data: Desc::Data,
    ) -> DataMut<'_, &'_ mut Desc::Data, S> {
        let ctxt = <S as ContextHolder>::reborrow_context(&mut self.context);
        Desc::add_data(&mut self.inner, name, data, ctxt)
    }

    pub(crate) fn add_attached_geometry(
        &'_ mut self,
        name: String,
        args: <Desc::Attached<S> as AttachedGeometry<S>>::Args,
        position: AttachmentPosition,
    ) -> DataMut<'_, &'_ mut Desc::Attached<S>, S> {
        let ctxt = <S as ContextHolder>::reborrow_context(&mut self.context);
        Desc::add_attached_geometry(&mut self.inner, name, args, position, ctxt)
    }

    pub(crate) fn update_settings(&mut self, rebuild_pipeline: bool) -> &mut Self {
        Desc::update_settings(&mut self.inner, &mut self.context, rebuild_pipeline);
        self
    }
}
