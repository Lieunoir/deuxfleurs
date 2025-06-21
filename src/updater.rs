use std::mem::transmute;
use std::ops::Deref;

use crate::aabb::SBV;
use crate::attachment::{NewVectorField, VectorField};
use crate::camera::Camera;
use crate::data::{DataSettings, DataUniform, DataUniformBuilder, TransformSettings};
use crate::ui::UiDataElement;
use egui::{SliderClamping, Widget};
use indexmap::IndexMap;

// Picker now uses same geometry as base
//
// Problems: mut ref in mut ver implies only one non mut borrow at a time
//   how to make sure data uniform modification -> only given when created for now, so not yet built

pub struct GraphicalContext<'a> {
    pub(crate) settings: &'a crate::Settings,
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) camera_light_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(crate) counter_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(crate) color_format: wgpu::TextureFormat,
    pub(crate) refresh_screen: &'a mut bool,
}

// `Renderer` can be `()` !
pub struct Element<Geometry, Renderer, Settings, Data, AttachedGeometry> {
    name: String,
    pub(crate) geometry: Geometry,
    pub(crate) renderer: Renderer,
    pub(crate) show: bool,
    pub(crate) transform: TransformSettings,
    pub(crate) settings: Settings,
    data: IndexMap<String, Data>,
    attached_data: IndexMap<String, AttachedGeometry>,
    shown_data: Option<String>,
    pub(crate) sbv: SBV,
}

pub type UninitedElement<Geometry, Settings, Data, AttachedGeometry> =
    Element<Geometry, (), Settings, Data, AttachedGeometry>;

pub type DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedGeometry> =
    Element<Geometry, Renderer<Fixed, DataB, Pipeline>, Settings, Data, AttachedGeometry>;

//mod private {
//    pub trait Sealed {}
//}

pub(crate) trait ElementTrait<'a> {
    //type Geometry: ElementGeometry;
    //type Settings: NamedSettings;
    //type Fixed: FixedRenderer<
    //    Settings = Self::Settings,
    //    Data = Self::Data,
    //    Geometry = Self::Geometry,
    //>;
    //type DataB: DataBuffer<Settings = Self::Settings, Data = Self::Data, Geometry = Self::Geometry>;
    //type Pipeline: RenderPipeline<
    //    Settings = Self::Settings,
    //    Data = Self::Data,
    //    Geometry = Self::Geometry,
    //    Fixed = Self::Fixed,
    //>;
    //type Data: DataUniformBuilder + DataSettings + UiDataElement;
    //type AttachedGeometry: AttachedGeometry;
    type Data;
    type Context;

    fn show(&mut self, show: bool, context: &mut Self::Context);
    //{
    //    if self.element.show != show {
    //        *self.context.refresh_screen = true;
    //        self.element.show = show;
    //    }
    //    self
    //}

    fn set_data(&mut self, name: Option<String>, context: &mut Self::Context);

    fn add_data(
        &mut self,
        name: String,
        data: Self::Data,
        context: &mut Self::Context,
    ) -> &mut Self::Data;

    fn update_settings(&mut self, context: &mut Self::Context, rebuild_pipeline: bool) {}
    //    self.settings
    //        .refresh_buffer(queue, &self.renderer.settings_uniform);
    //}

    fn update_transform(&mut self, context: &mut Self::Context) {}
    //    self.transform
    //        .to_raw()
    //        .refresh_buffer(queue, &self.renderer.transform_uniform);
    //}
    //{

    //type BareElement = Element<Geometry, (), Settings, Data, NewVectorField>;
    //type DisplayElement = Element<Geometry, (), Settings, Data, NewVectorField>;
}

impl<'a, Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedG> ElementTrait<'a>
    for DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedG>
where
    Geometry: ElementGeometry,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
    Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
    DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
    //AttachedG: AttachedGeometry,
{
    //type Geometry = Geometry;
    //type Settings = Settings;
    //type Fixed = Fixed;
    //type DataB = DataB;
    //type Pipeline = Pipeline;
    type Data = Data;
    //type AttachedGeometry = AttachedG;
    type Context = GraphicalContext<'a>;

    fn show(&mut self, show: bool, context: &mut Self::Context) {
        if self.show != show {
            *context.refresh_screen = true;
            self.show = show;
        }
    }

    fn set_data(&mut self, name: Option<String>, context: &mut Self::Context) {
        if self.shown_data != name {
            self.shown_data = name;
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            self.renderer
                .build_data_buffer(context.device, &self.geometry, data);
            data.map(|d| {
                self.renderer
                    .set_data_uniform(d.build_uniform(context.device))
            });
            self.renderer.build_pipeline(
                context.device,
                data,
                &self.settings,
                context.camera_light_bind_group_layout,
                context.color_format,
            );
            *context.refresh_screen = true;
        }
    }

    fn add_data(
        &mut self,
        name: String,
        data: Self::Data,
        context: &mut Self::Context,
    ) -> &mut Data {
        let old_data = self.data.insert(name.clone(), data);
        let data = self.data.get_mut(&name).unwrap();
        old_data.map(|old| data.apply_settings(old));
        if self.shown_data.as_ref() == Some(&name) {
            self.renderer
                .build_data_buffer(context.device, &self.geometry, Some(data));
            self.renderer
                .set_data_uniform(data.build_uniform(context.device));
            // Previously shown data can have same name but different type, thus requiring pipeline rebuild
            self.renderer.build_pipeline(
                context.device,
                Some(data),
                &self.settings,
                context.camera_light_bind_group_layout,
                context.color_format,
            );
            *context.refresh_screen = true;
        }
        data
    }

    fn update_settings(&mut self, context: &mut Self::Context, rebuild_pipeline: bool) {
        self.settings
            .refresh_buffer(context.queue, &self.renderer.settings_uniform);
        if rebuild_pipeline {
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            self.renderer.build_pipeline(
                context.device,
                data,
                &self.settings,
                context.camera_light_bind_group_layout,
                context.color_format,
            );
        }
    }

    fn update_transform(&mut self, context: &mut Self::Context) {
        self.transform
            .to_raw()
            .refresh_buffer(context.queue, &self.renderer.transform_uniform);
    }
}

impl<'a, Geometry, Settings, Data, AttachedG> ElementTrait<'a>
    for UninitedElement<Geometry, Settings, Data, AttachedG>
where
    Geometry: ElementGeometry,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
    //AttachedG: AttachedGeometry,
{
    //type AttachedGeometry = AttachedG;
    type Data = Data;
    type Context = ();

    fn show(&mut self, show: bool, context: &mut Self::Context) {
        self.show = show;
    }

    fn set_data(&mut self, name: Option<String>, context: &mut Self::Context) {
        self.shown_data = name;
    }

    fn add_data(
        &mut self,
        name: String,
        data: Self::Data,
        context: &mut Self::Context,
    ) -> &mut Data {
        let old_data = self.data.insert(name.clone(), data);
        let data = self.data.get_mut(&name).unwrap();
        old_data.map(|old| data.apply_settings(old));
        data
    }
}

// Upgrade Using AttachedGeometry::initialize or smth

pub struct ElementMut<'a, Element, Context> {
    pub(crate) element: &'a mut Element,
    pub(crate) context: Context,
}

impl<'a, Element, Context> Deref for ElementMut<'a, Element, Context> {
    type Target = Element;

    fn deref(&self) -> &Self::Target {
        self.element
    }
}

impl<Geometry, Renderer, Settings, Data, AttachedGeometry>
    Element<Geometry, Renderer, Settings, Data, AttachedGeometry>
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

    pub fn get_data(&self, name: &str) -> Option<&Data> {
        self.data.get(name)
    }

    pub fn get_attached_geometry(&self, name: &str) -> Option<&AttachedGeometry> {
        self.attached_data.get(name)
    }
}

impl<Geometry, Settings, Data, Attached> UninitedElement<Geometry, Settings, Data, Attached>
where
    Geometry: ElementGeometry,
    Settings: DataUniformBuilder + NamedSettings,
    Attached: AttachedGeometry,
{
    pub(crate) fn new_bare(name: String, args: Geometry::Args) -> Self {
        let geometry = Geometry::new(args);
        let transform = TransformSettings::default();
        let settings = Settings::default().set_name(&name);
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
        }
    }

    pub(crate) fn upgrade<Fixed, DataB, Pipeline>(
        self,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Element<
        Geometry,
        Renderer<Fixed, DataB, Pipeline>,
        Settings,
        Data,
        Attached::NewAttachedGeometry,
    >
    where
        Data: DataUniformBuilder,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
        Pipeline:
            RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
    {
        let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
        let renderer = Renderer::new(
            device,
            &self.geometry,
            &self.transform,
            &self.settings,
            data,
            camera_light_bind_group_layout,
            color_format,
        );

        let attached_data = self
            .attached_data
            .into_iter()
            .map(|(name, field)| {
                (
                    name,
                    field.init(
                        device,
                        camera_light_bind_group_layout,
                        &renderer.transform_uniform.bind_group_layout,
                        color_format,
                    ),
                )
            })
            .collect();

        Element {
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
        }
    }
}

impl<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedGeometry>
    DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedGeometry>
where
    Geometry: ElementGeometry,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
    Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
    DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
{
    pub(crate) fn new(
        name: String,
        args: Geometry::Args,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let geometry = Geometry::new(args);
        let transform = TransformSettings::default();
        let settings = Settings::default().set_name(&name);
        let renderer = Renderer::new(
            device,
            &geometry,
            &transform,
            &settings,
            None,
            camera_light_bind_group_layout,
            color_format,
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
        }
    }

    pub(crate) fn add_data(
        &mut self,
        name: String,
        data: Data,
        device: &wgpu::Device,
        //camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        //color_format: wgpu::TextureFormat,
        refresh_screen: &mut bool,
    ) -> &mut Data {
        let old_data = self.data.insert(name.clone(), data);
        let data = self.data.get_mut(&name).unwrap();
        old_data.map(|old| data.apply_settings(old));
        if self.shown_data.as_ref() == Some(&name) {
            self.renderer
                .build_data_buffer(device, &self.geometry, Some(data));
            self.renderer.set_data_uniform(data.build_uniform(device));
            //self.renderer.build_pipeline(
            //    device,
            //    Some(data),
            //    &self.settings,
            //    camera_light_bind_group_layout,
            //    color_format,
            //);
            *refresh_screen = true;
        }
        data
    }

    //pub(crate) fn set_data(
    //    &mut self,
    //    name: Option<String>,
    //    device: &wgpu::Device,
    //    camera_light_bind_group_layout: &wgpu::BindGroupLayout,
    //    color_format: wgpu::TextureFormat,
    //    refresh_screen: &mut bool,
    //) {
    //    if self.shown_data != name {
    //        self.shown_data = name;
    //        let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
    //        self.renderer
    //            .build_data_buffer(device, &self.geometry, data);
    //        data.map(|d| self.renderer.set_data_uniform(d.build_uniform(device)));
    //        self.renderer.build_pipeline(
    //            device,
    //            data,
    //            &self.settings,
    //            camera_light_bind_group_layout,
    //            color_format,
    //        );
    //        *refresh_screen = true;
    //    }
    //}

    //fn update_settings(&mut self, queue: &wgpu::Queue) {
    //    self.settings
    //        .refresh_buffer(queue, &self.renderer.settings_uniform);
    //}

    //fn update_transform(&mut self, queue: &wgpu::Queue) {
    //    self.transform
    //        .to_raw()
    //        .refresh_buffer(queue, &self.renderer.transform_uniform);
    //}

    pub(crate) fn draw_ui(
        &mut self,
        ui: &mut egui::Ui,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
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
                self.renderer.build_pipeline(
                    device,
                    data,
                    &self.settings,
                    camera_light_bind_group_layout,
                    color_format,
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
                        let mut prev_active = active;
                        ui.checkbox(&mut prev_active, name.clone());
                        if prev_active != active {
                            let data = if !active {
                                self.shown_data = Some(name.clone());
                                Some(&*data)
                            } else {
                                self.shown_data = None;
                                None
                            };
                            self.renderer
                                .build_data_buffer(device, &self.geometry, data);
                            self.renderer
                                .set_data_uniform(data.map(|d| d.build_uniform(device)).flatten());
                            self.renderer.build_pipeline(
                                device,
                                data,
                                &self.settings,
                                camera_light_bind_group_layout,
                                color_format,
                            );
                            *refresh_screen = true;
                        }
                    })
                })
                .body(|ui| {
                    if data.draw_ui(ui) && active {
                        if let Some(data_uniform) = self.renderer.get_data_uniform() {
                            data.refresh_buffer(queue, data_uniform);
                            *refresh_screen = true;
                        }
                    }
                });
        }

        //for (name, field) in &mut self.attached_data {
        //    let id = ui.make_persistent_id(name);
        //    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
        //        .show_header(ui, |ui| {
        //            ui.horizontal(|ui| {
        //                if ui
        //                    .checkbox(&mut field.settings.show, name.clone())
        //                    .changed()
        //                {
        //                    self.dirty = true;
        //                }
        //            });
        //        })
        //        .body(|ui| {
        //            //TODO move this
        //            if egui::Slider::new(&mut field.settings.magnitude, 0.1..=100.0)
        //                .text("Magnitude")
        //                .clamping(SliderClamping::Never)
        //                .logarithmic(true)
        //                .ui(ui)
        //                .changed()
        //            {
        //                field.settings_changed = true;
        //            }

        //            field.settings_changed |= field.settings.color.draw(ui, &mut false);
        //        });
        //}
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

    fn render_attached_data<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        //for (_, attached) in &self.attached_data {
        //    attached.render(render_pass);
        //}
    }
}

impl<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedGeometry> Render
    for DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedGeometry>
where
    Geometry: ElementGeometry,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
    Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
    DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
    Renderer<Fixed, DataB, Pipeline>: Render,
{
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show {
            self.renderer.render(render_pass);
            self.render_attached_data(render_pass);
        }
    }

    fn render_shadow<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show {
            self.renderer.render_shadow(render_pass);
        }
    }

    fn render_picker<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show {
            self.renderer.render_picker(render_pass);
        }
    }
}

//impl<'a, Geometry, Renderer, Settings, Data, AttachedGeometry>
//    ElementMut<'a, Element<Geometry, Renderer, Settings, Data, AttachedGeometry>, ()>
//{
//    pub fn show(self, show: bool) -> Self {
//        self.element.show = show;
//        self
//    }
//
//    pub fn set_data(self, name: Option<String>) -> Self {
//        self.element.shown_data = name;
//        self
//    }
//}

impl<'a, Geometry, Renderer, Settings, Data, AttachedGeometry, Context>
    ElementMut<'a, Element<Geometry, Renderer, Settings, Data, AttachedGeometry>, Context>
where
    Element<Geometry, Renderer, Settings, Data, AttachedGeometry>:
        ElementTrait<'a, Context = Context, Data = Data>,
{
    pub fn show(&mut self, show: bool) -> &mut Self {
        self.element.show(show, &mut self.context);
        self
    }

    pub fn set_data(&mut self, name: Option<String>) -> &mut Self {
        self.element.set_data(name, &mut self.context);
        self
    }

    pub(crate) fn add_data(&mut self, name: String, data: Data) -> &mut Data {
        self.element.add_data(name, data, &mut self.context)
    }

    pub(crate) fn update_settings(&mut self, rebuild_pipeline: bool) -> &mut Self {
        self.element
            .update_settings(&mut self.context, rebuild_pipeline);
        self
    }

    pub(crate) fn update_transform(&mut self) -> &mut Self {
        self.element.update_transform(&mut self.context);
        self
    }
}

//impl<'a, Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedGeometry>
//    ElementMut<
//        'a,
//        Element<Geometry, Renderer<Fixed, DataB, Pipeline>, Settings, Data, AttachedGeometry>,
//        GraphicalContext<'a>,
//    >
//where
//    Geometry: ElementGeometry,
//    Data: DataUniformBuilder + DataSettings + UiDataElement,
//    Settings: NamedSettings,
//    Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
//    DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
//    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
//{
//    pub fn show(self, show: bool) -> Self {
//        if self.element.show != show {
//            *self.context.refresh_screen = true;
//            self.element.show = show;
//        }
//        self
//    }
//
//    pub fn set_data(self, name: Option<String>) -> Self {
//        self.element.set_data(
//            name,
//            self.context.device,
//            self.context.camera_light_bind_group_layout,
//            self.context.color_format,
//            self.context.refresh_screen,
//        );
//        self
//    }
//
//    fn update_settings(self) -> Self {
//        self.element.update_settings(self.context.queue);
//        self
//    }
//
//    fn update_transform(self) -> Self {
//        self.element.update_transform(self.context.queue);
//        self
//    }
//}

pub(crate) trait AttachedGeometry {
    type NewAttachedGeometry;

    fn init(
        self,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self::NewAttachedGeometry;
}

impl AttachedGeometry for () {
    type NewAttachedGeometry = ();

    fn init(
        self,
        _device: &wgpu::Device,
        _camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        _transform_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
    ) -> Self::NewAttachedGeometry {
    }
}

pub(crate) trait NamedSettings: Default + DataUniformBuilder {
    fn set_name(self, name: &str) -> Self;

    fn draw_ui(&mut self, ui: &mut egui::Ui, rebuild_pipeline: &mut bool) -> bool;
}

pub struct Renderer<Fixed, DataB, Pipeline> {
    pub(crate) fixed: Fixed,
    pub(crate) data_buffer: DataB,
    pub(crate) pipeline: Pipeline,
    pub(crate) transform_uniform: DataUniform,
    pub(crate) settings_uniform: DataUniform,
    pub(crate) data_uniform: Option<DataUniform>,
}

impl<
        Settings: DataUniformBuilder,
        Data: DataUniformBuilder,
        Geometry,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
    > Renderer<Fixed, DataB, Pipeline>
{
    pub(crate) fn new(
        device: &wgpu::Device,
        geometry: &Geometry,
        transform: &TransformSettings,
        settings: &Settings,
        data: Option<&Data>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let fixed = Fixed::initialize(device, geometry);
        let data_buffer = DataB::new(device, geometry, data);

        let transform_uniform = transform.to_raw().build_uniform(device).unwrap();
        let settings_uniform = settings.build_uniform(device).unwrap();
        let data_uniform = data.map(|d| d.build_uniform(device)).flatten();
        let pipeline = RenderPipeline::new(
            device,
            data,
            &fixed,
            settings,
            &transform_uniform,
            &settings_uniform,
            data_uniform.as_ref(),
            camera_light_bind_group_layout,
            color_format,
        );
        //TODO can be factored
        Self {
            fixed,
            data_buffer,
            pipeline,
            transform_uniform,
            settings_uniform,
            data_uniform,
        }
    }

    fn set_data_uniform(&mut self, data_uniform: Option<DataUniform>) {
        self.data_uniform = data_uniform;
    }

    fn get_data_uniform(&mut self) -> Option<&DataUniform> {
        self.data_uniform.as_ref()
    }

    fn update_settings(&mut self, settings: &Settings, queue: &wgpu::Queue) {
        settings.refresh_buffer(queue, &self.settings_uniform);
    }

    fn build_data_buffer(
        &mut self,
        device: &wgpu::Device,
        geometry: &Geometry,
        data: Option<&Data>,
    ) {
        self.data_buffer = DataB::new(device, geometry, data);
    }

    fn build_pipeline(
        &mut self,
        device: &wgpu::Device,
        data: Option<&Data>,
        settings: &Settings,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) {
        self.pipeline = RenderPipeline::new(
            device,
            data,
            &self.fixed,
            settings,
            &self.transform_uniform,
            &self.settings_uniform,
            self.data_uniform.as_ref(),
            camera_light_bind_group_layout,
            color_format,
        );
    }
}

pub(crate) trait FixedRenderer {
    type Settings;
    type Data;
    type Geometry;

    fn initialize(device: &wgpu::Device, geometry: &Self::Geometry) -> Self;
}

pub(crate) trait DataBuffer {
    type Settings;
    type Data;
    type Geometry;

    fn new(device: &wgpu::Device, geometry: &Self::Geometry, data: Option<&Self::Data>) -> Self;
}

pub(crate) trait RenderPipeline {
    type Settings;
    type Data;
    type Geometry;
    type Fixed;

    fn new(
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        fixed: &Self::Fixed,
        settings: &Self::Settings,
        tansform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self;
}

pub(crate) trait Render {
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b;

    fn render_shadow<'a, 'b>(&'a self, _render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
    }

    fn render_picker<'a, 'b>(&'a self, _render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
    }
}

pub(crate) trait ElementPicker: Render {
    type Geometry;
    type Settings: DataUniformBuilder;

    fn new(
        geometry: &Self::Geometry,
        settings: &Self::Settings,
        transform: &TransformSettings,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self;

    fn update_transform(&self, queue: &wgpu::Queue, transform: &TransformSettings);

    fn update_settings(&self, queue: &wgpu::Queue, settings: &Self::Settings);

    fn get_total_elements(&self) -> u32;

    fn get_element(
        &self,
        _geometry: &Self::Geometry,
        _transform: &TransformSettings,
        _camera: &Camera,
        item: u32,
        _pos_x: f32,
        _pos_y: f32,
    ) -> u32 {
        item
    }
}

pub trait ElementGeometry {
    type Args;

    fn new(args: Self::Args) -> Self;

    fn get_positions(&self) -> &[[f32; 3]];

    fn get_total_elements(&self) -> u32;
}
