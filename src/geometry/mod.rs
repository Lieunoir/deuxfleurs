use crate::aabb::SBV;
use crate::data::TransformSettings;
use crate::data::internal::{DataSettings, DataUniform, DataUniformBuilder};
use crate::ui::UiDataElement;
pub(crate) use data::{DataMut, DataMutTrait};
use indexmap::IndexMap;
pub(crate) use renderer::*;
use std::ops::Deref;
mod data;
mod renderer;

pub trait Context {
    type DataUniform<'a>;
    type TransformLayout;
}

pub struct GraphicalContext<'a> {
    pub(crate) settings: &'a crate::Settings,
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) camera_light_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(crate) counter_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub(crate) color_format: wgpu::TextureFormat,
    pub(crate) refresh_screen: &'a mut bool,
}

impl<'a> Context for GraphicalContext<'a> {
    type DataUniform<'b> = &'b Option<DataUniform>;
    type TransformLayout = wgpu::BindGroupLayout;
}

impl Context for () {
    type DataUniform<'b> = ();
    type TransformLayout = ();
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

pub trait ElementTrait<Ctxt: Context> {
    type Data;
    type Geometry: ElementGeometry;
    type Attached: AttachedGeometry<Ctxt>;

    fn replace(&mut self, args: <Self::Geometry as ElementGeometry>::Args, context: &mut Ctxt);

    fn show(&mut self, show: bool, context: &mut Ctxt);

    fn set_data(&mut self, name: Option<String>, context: &mut Ctxt);

    fn add_data<'b>(
        &'b mut self,
        name: String,
        data: Self::Data,
        context: &'b mut Ctxt,
    ) -> DataMut<'b, Self::Data, Ctxt>;

    fn update_settings(&mut self, _context: &mut Ctxt, _rebuild_pipeline: bool) {}

    fn update_transform(&mut self, _context: &mut Ctxt) {}

    fn add_attached_geometry<'b>(
        &'b mut self,
        name: String,
        args: <Self::Attached as AttachedGeometry<Ctxt>>::Args,
        context: &'b mut Ctxt,
    ) -> DataMut<'b, Self::Attached, Ctxt>;
}

impl<'a, Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedG>
    ElementTrait<GraphicalContext<'a>>
    for DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedG>
where
    for<'b> AttachedG: AttachedGeometry<GraphicalContext<'b>>,
    Geometry: ElementGeometry,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
    Fixed: FixedRenderer<Geometry = Geometry>,
    DataB: DataBuffer<Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry>,
{
    type Data = Data;
    type Geometry = Geometry;
    type Attached = AttachedG;

    fn replace(
        &mut self,
        args: <Self::Geometry as ElementGeometry>::Args,
        context: &mut GraphicalContext<'_>,
    ) {
        let new_geometry = <Self::Geometry as ElementGeometry>::new(args);
        if self.geometry().can_be_replaced_by(&new_geometry) {
            self.renderer.fixed = Fixed::initialize(context.device, &new_geometry);
            self.geometry = new_geometry;
        } else {
            *self = Self::new_with_geometry(
                self.name.clone(),
                new_geometry,
                context.device,
                context.camera_light_bind_group_layout,
                context.counter_bind_group_layout,
                context.color_format,
            )
        }
    }

    fn show(&mut self, show: bool, context: &mut GraphicalContext<'_>) {
        if self.show != show {
            *context.refresh_screen = true;
            self.show = show;
        }
    }

    fn set_data(&mut self, name: Option<String>, context: &mut GraphicalContext<'_>) {
        if self.shown_data != name {
            self.shown_data = name;
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            self.renderer
                .build_data_buffer(context.device, &self.geometry, data);
            data.map(|d| {
                self.renderer
                    .set_data_uniform(d.build_uniform(context.device))
            });
            self.renderer.rebuild_pipeline(
                context.device,
                data,
                &self.settings,
                context.camera_light_bind_group_layout,
                context.color_format,
            );
            *context.refresh_screen = true;
        }
    }

    fn add_data<'b>(
        &'b mut self,
        name: String,
        data: Self::Data,
        context: &'b mut GraphicalContext<'a>,
    ) -> DataMut<'b, Self::Data, GraphicalContext<'a>> {
        let old_data = self.data.insert(name.clone(), data);
        let data = self.data.get_mut(&name).unwrap();
        old_data.map(|old| data.apply_settings(old));
        if self.shown_data.as_ref() == Some(&name) {
            self.renderer
                .build_data_buffer(context.device, &self.geometry, Some(data));
            self.renderer
                .set_data_uniform(data.build_uniform(context.device));
            // Previously shown data can have same name but different type, thus requiring pipeline rebuild
            self.renderer.rebuild_pipeline(
                context.device,
                Some(data),
                &self.settings,
                context.camera_light_bind_group_layout,
                context.color_format,
            );
            *context.refresh_screen = true;
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
                context.camera_light_bind_group_layout,
                context.color_format,
            );
        }
    }

    fn update_transform(&mut self, context: &mut GraphicalContext<'_>) {
        self.transform
            .to_raw()
            .refresh_buffer(context.queue, &self.renderer.transform_uniform);
    }

    fn add_attached_geometry<'b>(
        &'b mut self,
        name: String,
        args: <Self::Attached as AttachedGeometry<GraphicalContext<'a>>>::Args,
        context: &'b mut GraphicalContext<'a>,
    ) -> DataMut<'b, Self::Attached, GraphicalContext<'a>> {
        *context.refresh_screen = true;
        {
            let geometry = Self::Attached::new(
                name.clone(),
                args,
                context,
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

impl<Geometry, Settings, Data, AttachedG> ElementTrait<()>
    for UninitedElement<Geometry, Settings, Data, AttachedG>
where
    Geometry: ElementGeometry,
    AttachedG: AttachedGeometry<()>,
    Data: DataSettings,
    Settings: NamedSettings,
    AttachedG: NewAttachedGeometry,
{
    type Geometry = Geometry;
    type Attached = AttachedG;
    type Data = Data;

    fn replace(&mut self, args: <Self::Geometry as ElementGeometry>::Args, _context: &mut ()) {
        let new_geometry = <Self::Geometry as ElementGeometry>::new(args);
        if self.geometry().can_be_replaced_by(&new_geometry) {
            self.geometry = new_geometry;
        } else {
            *self = Self::new_bare_with_geometry(self.name.clone(), new_geometry)
        }
    }

    fn show(&mut self, show: bool, _context: &mut ()) {
        self.show = show;
    }

    fn set_data(&mut self, name: Option<String>, _context: &mut ()) {
        self.shown_data = name;
    }

    fn add_data<'b>(
        &'b mut self,
        name: String,
        data: Self::Data,
        context: &'b mut (),
    ) -> DataMut<'b, Self::Data, ()> {
        let old_data = self.data.insert(name.clone(), data);
        let data = self.data.get_mut(&name).unwrap();
        old_data.map(|old| data.apply_settings(old));
        DataMut {
            inner: data,
            uniform: (),
            context: context,
        }
    }

    fn add_attached_geometry<'b>(
        &'b mut self,
        name: String,
        args: <Self::Attached as AttachedGeometry<()>>::Args,
        context: &'b mut (),
    ) -> DataMut<'b, Self::Attached, ()> {
        let geometry = AttachedG::new(name.clone(), args, &mut (), &());
        self.attached_data.insert(name.clone(), geometry);
        DataMut {
            inner: self.attached_data.get_mut(&name).unwrap(),
            uniform: (),
            context: context,
        }
    }
}

pub struct ElementMut<'a, Element, Context> {
    pub(crate) element: &'a mut Element,
    pub(crate) context: Context,
}

impl<'a, Element: ElementTrait<Ctxt>, Ctxt: Context> Deref for ElementMut<'a, Element, Ctxt> {
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
    Attached: NewAttachedGeometry,
{
    pub(crate) fn new_bare(name: String, args: Geometry::Args) -> Self {
        let geometry = Geometry::new(args);
        Self::new_bare_with_geometry(name, geometry)
    }

    pub(crate) fn new_bare_with_geometry(name: String, geometry: Geometry) -> Self {
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
        counter_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Element<
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
            camera_light_bind_group_layout,
            counter_bind_group_layout,
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

impl<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>
    DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>
where
    for<'a> Attached: AttachedGeometry<GraphicalContext<'a>>,
    Geometry: ElementGeometry,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
    Fixed: FixedRenderer<Geometry = Geometry>,
    DataB: DataBuffer<Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry>,
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
        Self::new_with_geometry(
            name,
            geometry,
            device,
            camera_light_bind_group_layout,
            counter_bind_group_layout,
            color_format,
        )
    }

    pub(crate) fn new_with_geometry(
        name: String,
        geometry: Geometry,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let transform = TransformSettings::default();
        let settings = Settings::default().set_name(&name);
        let renderer = Renderer::new(
            device,
            &geometry,
            &transform,
            &settings,
            None,
            camera_light_bind_group_layout,
            counter_bind_group_layout,
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
                self.renderer.rebuild_pipeline(
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
                            self.renderer.rebuild_pipeline(
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
                        camera_light_bind_group_layout,
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

    fn render_attached_data<'c, 'd>(&'c self, render_pass: &mut wgpu::RenderPass<'d>)
    where
        'c: 'd,
    {
        for (_, attached) in &self.attached_data {
            attached.render(render_pass);
        }
    }
}

impl<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached> Render
    for DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>
where
    for<'a> Attached: AttachedGeometry<GraphicalContext<'a>>,
    Geometry: ElementGeometry,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
    Fixed: FixedRenderer<Geometry = Geometry>,
    DataB: DataBuffer<Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry>,
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

impl<'a, Element: ElementTrait<Ctxt>, Ctxt: Context> ElementMut<'a, Element, Ctxt> {
    pub fn show(&mut self, show: bool) -> &mut Self {
        self.element.show(show, &mut self.context);
        self
    }

    pub fn set_data(&mut self, name: Option<String>) -> &mut Self {
        self.element.set_data(name, &mut self.context);
        self
    }

    pub(crate) fn add_data(
        &mut self,
        name: String,
        data: Element::Data,
    ) -> DataMut<'_, Element::Data, Ctxt> {
        self.element.add_data(name, data, &mut self.context)
    }

    pub(crate) fn add_attached_geometry(
        &mut self,
        name: String,
        args: <Element::Attached as AttachedGeometry<Ctxt>>::Args,
    ) -> DataMut<'_, Element::Attached, Ctxt> {
        self.element
            .add_attached_geometry(name, args, &mut self.context)
    }

    pub(crate) fn update_settings(&mut self, rebuild_pipeline: bool) -> &mut Self {
        self.element
            .update_settings(&mut self.context, rebuild_pipeline);
        self
    }
}

pub trait AttachedGeometry<Ctxt: Context> {
    type Args;
    type Settings;

    fn new(
        name: String,
        args: Self::Args,
        context: &mut Ctxt,
        transform_layout: &Ctxt::TransformLayout,
    ) -> Self;

    fn shown(&self) -> bool {
        false
    }

    fn show(&mut self, _show: bool, _refresh_screen: &mut bool) {}

    fn draw_ui(
        &mut self,
        _ui: &mut egui::Ui,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
        _refresh_screen: &mut bool,
    ) {
    }

    fn render<'c, 'd>(&'c self, _render_pass: &mut wgpu::RenderPass<'d>)
    where
        'c: 'd,
    {
    }

    fn get_settings(&mut self) -> &mut Self::Settings;
}

pub trait NewAttachedGeometry {
    type UpgradedAttachedGeometry;

    fn init(
        self,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self::UpgradedAttachedGeometry;
}

impl<Ctxt: Context> AttachedGeometry<Ctxt> for () {
    type Args = ();
    type Settings = ();

    fn new(
        _name: String,
        _args: Self::Args,
        _context: &mut Ctxt,
        _transform_layout: &Ctxt::TransformLayout,
    ) -> Self {
        ()
    }

    fn get_settings(&mut self) -> &mut Self::Settings {
        self
    }
}

pub struct EmptyAttached(());

impl NewAttachedGeometry for () {
    type UpgradedAttachedGeometry = EmptyAttached;

    fn init(
        self,
        _device: &wgpu::Device,
        _camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        _transform_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
    ) -> Self::UpgradedAttachedGeometry {
        EmptyAttached(())
    }
}

impl<Ctxt: Context> AttachedGeometry<Ctxt> for EmptyAttached {
    type Args = ();
    type Settings = ();

    fn new(
        _name: String,
        _args: Self::Args,
        _context: &mut Ctxt,
        _transform_layout: &Ctxt::TransformLayout,
    ) -> Self {
        EmptyAttached(())
    }

    fn get_settings(&mut self) -> &mut Self::Settings {
        &mut self.0
    }
}

pub trait NamedSettings: Default + DataUniformBuilder {
    fn set_name(self, name: &str) -> Self;

    fn draw_ui(&mut self, ui: &mut egui::Ui, rebuild_pipeline: &mut bool) -> bool;
}

pub trait ElementGeometry {
    type Args;

    fn new(args: Self::Args) -> Self;

    fn can_be_replaced_by(&self, _other: &Self) -> bool {
        false
    }

    fn get_positions(&self) -> &[[f32; 3]];

    fn get_total_elements(&self) -> u32;
}
