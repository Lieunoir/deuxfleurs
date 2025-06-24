use crate::aabb::SBV;
use crate::data::{DataSettings, DataUniform, DataUniformBuilder, TransformSettings};
use crate::ui::UiDataElement;
use indexmap::IndexMap;
use std::ops::Deref;

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

pub trait ElementTrait {
    type Data;
    type Context<'a>;
    type DataMutUniform<'a>
    where
        Self: 'a;
    type Attached: AttachedGeometry;

    fn show(&mut self, show: bool, context: &mut Self::Context<'_>);

    fn set_data(&mut self, name: Option<String>, context: &mut Self::Context<'_>);

    fn add_data<'a, 'b>(
        &'b mut self,
        name: String,
        data: Self::Data,
        context: &'b mut Self::Context<'a>,
    ) -> DataMut<'b, Self::Data, &'b mut Self::Context<'a>, Self::DataMutUniform<'b>>;

    fn update_settings(&mut self, _context: &mut Self::Context<'_>, _rebuild_pipeline: bool) {}

    fn update_transform(&mut self, _context: &mut Self::Context<'_>) {}

    fn add_attached_geometry<'a, 'b>(
        &'b mut self,
        name: String,
        args: <Self::Attached as AttachedGeometry>::Args,
        context: &'b mut Self::Context<'a>,
    ) -> DataMut<'b, Self::Attached, &'b mut Self::Context<'a>, Self::DataMutUniform<'b>>;
}

impl<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedG> ElementTrait
    for DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, AttachedG>
where
    for<'a> AttachedG: AttachedGeometry<
        Context<'a> = GraphicalContext<'a>,
        TransformLayout = wgpu::BindGroupLayout,
    >,
    Geometry: ElementGeometry,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
    Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
    DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
{
    type Data = Data;
    type Context<'a> = GraphicalContext<'a>;
    type DataMutUniform<'a>
        = &'a Option<DataUniform>
    where
        AttachedG: 'a,
        Geometry: 'a,
        Fixed: 'a,
        DataB: 'a,
        Pipeline: 'a,
        Settings: 'a,
        Data: 'a;
    type Attached = AttachedG;

    fn show(&mut self, show: bool, context: &mut Self::Context<'_>) {
        if self.show != show {
            *context.refresh_screen = true;
            self.show = show;
        }
    }

    fn set_data(&mut self, name: Option<String>, context: &mut Self::Context<'_>) {
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

    fn add_data<'a, 'b>(
        &'b mut self,
        name: String,
        data: Self::Data,
        context: &'b mut Self::Context<'a>,
    ) -> DataMut<'b, Self::Data, &'b mut Self::Context<'a>, Self::DataMutUniform<'b>> {
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

    fn update_settings(&mut self, context: &mut Self::Context<'_>, rebuild_pipeline: bool) {
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

    fn update_transform(&mut self, context: &mut Self::Context<'_>) {
        self.transform
            .to_raw()
            .refresh_buffer(context.queue, &self.renderer.transform_uniform);
    }

    fn add_attached_geometry<'a, 'b>(
        &'b mut self,
        name: String,
        args: <Self::Attached as AttachedGeometry>::Args,
        context: &'b mut Self::Context<'a>,
    ) -> DataMut<'b, Self::Attached, &'b mut Self::Context<'a>, Self::DataMutUniform<'b>> {
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

impl<Geometry, Settings, Data, AttachedG> ElementTrait
    for UninitedElement<Geometry, Settings, Data, AttachedG>
where
    Geometry: ElementGeometry,
    for<'a> AttachedG: AttachedGeometry<Context<'a> = &'a (), TransformLayout = ()>,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
{
    type Attached = AttachedG;
    type Data = Data;
    type Context<'a> = ();
    type DataMutUniform<'a>
        = ()
    where
        AttachedG: 'a,
        Geometry: 'a,
        Settings: 'a,
        Data: 'a;

    fn show(&mut self, show: bool, _context: &mut Self::Context<'_>) {
        self.show = show;
    }

    fn set_data(&mut self, name: Option<String>, _context: &mut Self::Context<'_>) {
        self.shown_data = name;
    }

    fn add_data<'a, 'b>(
        &'b mut self,
        name: String,
        data: Self::Data,
        context: &'b mut Self::Context<'a>,
    ) -> DataMut<'b, Self::Data, &'b mut Self::Context<'a>, Self::DataMutUniform<'b>> {
        let old_data = self.data.insert(name.clone(), data);
        let data = self.data.get_mut(&name).unwrap();
        old_data.map(|old| data.apply_settings(old));
        DataMut {
            inner: data,
            uniform: (),
            context: context,
        }
    }

    fn add_attached_geometry<'a, 'b>(
        &'b mut self,
        name: String,
        args: <Self::Attached as AttachedGeometry>::Args,
        context: &'b mut Self::Context<'a>,
    ) -> DataMut<'b, Self::Attached, &'b mut Self::Context<'a>, Self::DataMutUniform<'b>> {
        let geometry = AttachedG::new(name.clone(), args, &mut &(), &());
        self.attached_data.insert(name.clone(), geometry);
        DataMut {
            inner: self.attached_data.get_mut(&name).unwrap(),
            uniform: (),
            context: context,
        }
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
    Attached: NewAttachedGeometry,
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

impl<'a, 'b, Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>
    DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>
where
    'a: 'b,
    Attached: AttachedGeometry,
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
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
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
    Attached: AttachedGeometry,
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

pub struct DataMut<'a, T, Context, Uniform> {
    pub(crate) inner: &'a mut T,
    uniform: Uniform,
    context: Context,
}

pub type UninitedData<'a, 'b, T> = DataMut<'a, T, &'b mut (), ()>;
pub type DisplayData<'a, 'b, T>
where
    T: DataUniformBuilder,
= DataMut<'a, T, &'b mut GraphicalContext<'a>, &'b mut Option<DataUniform>>;

impl<'a, T, Context, Uniform> DataMut<'a, T, Context, Uniform> {
    pub(crate) fn convert<U, F: FnOnce(&mut T) -> &mut U>(
        self,
        f: F,
    ) -> DataMut<'a, U, Context, Uniform> {
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

impl<'a, 'b, T> DataMutTrait for UninitedData<'a, 'b, T> {
    fn update_data_settings(&mut self) {}
}

impl<'a, 'b, T> DataMutTrait for DisplayData<'a, 'b, T>
where
    T: DataUniformBuilder,
{
    fn update_data_settings(&mut self) {
        self.uniform
            .as_ref()
            .map(|uniform| self.inner.refresh_buffer(self.context.queue, uniform));
    }
}

impl<'a, Geometry, Renderer, Settings, Data, Attached, Context>
    ElementMut<'a, Element<Geometry, Renderer, Settings, Data, Attached>, Context>
where
    Element<Geometry, Renderer, Settings, Data, Attached>:
        ElementTrait<Context<'a> = Context, Data = Data>,
{
    pub fn show(&mut self, show: bool) -> &mut Self {
        self.element.show(show, &mut self.context);
        self
    }

    pub fn set_data(&mut self, name: Option<String>) -> &mut Self {
        self.element.set_data(name, &mut self.context);
        self
    }

    pub(crate) fn add_data<'b, 'c>(
        &'c mut self,
        name: String,
        data: Data,
    ) -> DataMut<
        'b,
        Data,
        &'b mut <Element<Geometry, Renderer, Settings, Data, Attached> as ElementTrait>::Context<
            'a,
        >,
        <Element<Geometry, Renderer, Settings, Data, Attached> as ElementTrait>::DataMutUniform<'b>,
    >
    where
        'a: 'b,
        'c: 'b,
    {
        self.element.add_data(name, data, &mut self.context)
    }

    pub(crate) fn add_attached_geometry<'b>(
        &'b mut self,
        name: String,
        args: <<Element<Geometry, Renderer, Settings, Data, Attached> as ElementTrait>::Attached as AttachedGeometry>::Args,
    ) -> DataMut<
        'b,
        <Element<Geometry, Renderer, Settings, Data, Attached> as ElementTrait>::Attached,
        &'b mut <Element<Geometry, Renderer, Settings, Data, Attached> as ElementTrait>::Context<
            'a,
        >,
        <Element<Geometry, Renderer, Settings, Data, Attached> as ElementTrait>::DataMutUniform<'b>,
    >
    where
        'a: 'b,
    {
        self.element
            .add_attached_geometry(name, args, &mut self.context)
    }

    pub(crate) fn update_settings(&mut self, rebuild_pipeline: bool) -> &mut Self {
        self.element
            .update_settings(&mut self.context, rebuild_pipeline);
        self
    }
}

pub trait AttachedGeometry {
    type Args;
    type Context<'a>;
    type TransformLayout;
    type Settings;

    fn new<'a>(
        name: String,
        args: Self::Args,
        context: &mut Self::Context<'a>,
        transform_layout: &Self::TransformLayout,
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

impl AttachedGeometry for () {
    type Args = ();
    type Context<'a> = &'a ();
    type TransformLayout = ();
    type Settings = ();

    fn new(
        _name: String,
        _args: Self::Args,
        _context: &mut Self::Context<'_>,
        _transform_layout: &Self::TransformLayout,
    ) -> Self {
        ()
    }

    fn get_settings(&mut self) -> &mut Self::Settings {
        self
    }
}

impl NewAttachedGeometry for () {
    type UpgradedAttachedGeometry = ();

    fn init(
        self,
        _device: &wgpu::Device,
        _camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        _transform_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
    ) -> Self::UpgradedAttachedGeometry {
    }
}

pub trait NamedSettings: Default + DataUniformBuilder {
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
        counter_bind_group_layout: &wgpu::BindGroupLayout,
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
            geometry,
            settings,
            &transform_uniform,
            &settings_uniform,
            data_uniform.as_ref(),
            camera_light_bind_group_layout,
            counter_bind_group_layout,
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

    fn build_data_buffer(
        &mut self,
        device: &wgpu::Device,
        geometry: &Geometry,
        data: Option<&Data>,
    ) {
        self.data_buffer = DataB::new(device, geometry, data);
    }

    fn rebuild_pipeline(
        &mut self,
        device: &wgpu::Device,
        data: Option<&Data>,
        settings: &Settings,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) {
        self.pipeline.rebuild(
            device,
            data,
            settings,
            &self.transform_uniform,
            &self.settings_uniform,
            self.data_uniform.as_ref(),
            camera_light_bind_group_layout,
            color_format,
        );
    }
}

pub trait FixedRenderer {
    type Settings;
    type Data;
    type Geometry;

    fn initialize(device: &wgpu::Device, geometry: &Self::Geometry) -> Self;
}

pub trait DataBuffer {
    type Settings;
    type Data;
    type Geometry;

    fn new(device: &wgpu::Device, geometry: &Self::Geometry, data: Option<&Self::Data>) -> Self;
}

pub trait RenderPipeline {
    type Settings;
    type Data;
    type Geometry;
    type Fixed;

    fn new(
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        geometry: &Self::Geometry,
        settings: &Self::Settings,
        tansform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self;

    fn rebuild(
        &mut self,
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        settings: &Self::Settings,
        transform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    );
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

pub trait ElementGeometry {
    type Args;

    fn new(args: Self::Args) -> Self;

    fn get_positions(&self) -> &[[f32; 3]];

    fn get_total_elements(&self) -> u32;
}
