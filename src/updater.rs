use crate::aabb::SBV;
use crate::attachment::{NewVectorField, VectorField};
use crate::camera::Camera;
use crate::data::{DataSettings, DataUniform, DataUniformBuilder, TransformSettings};
use crate::ui::UiDataElement;
use egui::{SliderClamping, Widget};
use indexmap::IndexMap;

pub struct MainDisplayGeometry<
    Settings: NamedSettings,
    Data: DataUniformBuilder + UiDataElement + DataSettings,
    Geometry: Positions,
    Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
    DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
    Picker: ElementPicker<Geometry = Geometry, Settings = Settings>,
> where
    Renderer<Settings, Data, Geometry, Fixed, DataB, Pipeline>: Render,
{
    pub name: String,
    pub geometry: Geometry,
    pub(crate) sbv: SBV,
    pub(crate) show: bool,
    pub(crate) updater: Updater<Settings, Data>,
    pub(crate) renderer: Renderer<Settings, Data, Geometry, Fixed, DataB, Pipeline>,
    pub(crate) picker: Picker,
}

impl<
        Settings: NamedSettings,
        Data: DataUniformBuilder + UiDataElement + DataSettings,
        Geometry: Positions,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
        Picker: ElementPicker<Geometry = Geometry, Settings = Settings>,
    > Render for MainDisplayGeometry<Settings, Data, Geometry, Fixed, DataB, Pipeline, Picker>
where
    Renderer<Settings, Data, Geometry, Fixed, DataB, Pipeline>: Render,
{
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show {
            self.renderer.render(render_pass);
            self.updater.render_attached_data(render_pass);
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
}

impl<
        Settings: NamedSettings,
        Data: DataUniformBuilder + UiDataElement + DataSettings,
        Geometry: Positions,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
        Picker: ElementPicker<Geometry = Geometry, Settings = Settings>,
    > MainDisplayGeometry<Settings, Data, Geometry, Fixed, DataB, Pipeline, Picker>
where
    Renderer<Settings, Data, Geometry, Fixed, DataB, Pipeline>: Render,
{
    pub(crate) fn init(
        device: &wgpu::Device,
        name: String,
        geometry: Geometry,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let updater = Updater::new(&name);
        let renderer = Renderer::new(
            device,
            &geometry,
            &updater.transform,
            &updater.settings,
            camera_light_bind_group_layout,
            color_format,
        );
        let picker = Picker::new(
            &geometry,
            &updater.settings,
            &updater.transform,
            device,
            camera_light_bind_group_layout,
            counter_bind_group_layout,
        );
        let sbv = SBV::new(geometry.get_positions());

        Self {
            name,
            geometry,
            show: true,
            sbv,
            renderer,
            updater,
            picker,
        }
    }

    pub(crate) fn refresh(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        w_sbv: &mut Option<SBV>,
    ) -> bool {
        self.updater.refresh(
            &self.geometry,
            device,
            queue,
            camera_light_bind_group_layout,
            color_format,
            &mut self.renderer,
            &mut self.picker,
            self.show,
            &self.sbv,
            w_sbv,
        )
    }

    pub fn set_data(&mut self, name: Option<String>) -> &mut Self {
        self.updater.data_to_show = Some(name);
        self
    }

    pub(crate) fn draw_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.show, "Show").changed() {
                self.updater.dirty = true;
            }
        });
        self.updater.draw(ui, self.geometry.get_positions());
    }

    pub(crate) fn draw_gizmo(
        &mut self,
        ui: &mut egui::Ui,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
        gizmo_hover: &mut bool,
    ) {
        self.updater
            .draw_gizmo(ui, &self.name, view, proj, gizmo_hover);
    }

    pub(crate) fn render_picker<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show {
            self.picker.render(render_pass);
        }
    }

    pub fn get_total_elements(&self) -> u32 {
        self.picker.get_total_elements()
    }
}

pub(crate) trait NamedSettings: Default + DataUniformBuilder + UiDataElement {
    fn set_name(self, name: &str) -> Self;
}

pub(crate) struct Updater<Settings: NamedSettings, Data: DataUniformBuilder + UiDataElement> {
    pub(crate) transform: TransformSettings,
    pub(crate) settings: Settings,
    pub(crate) data: IndexMap<String, Data>,
    pub(crate) queued_attached_data: Vec<NewVectorField>,
    attached_data: IndexMap<String, VectorField>,
    pub(crate) shown_data: Option<String>,
    pub(crate) data_to_show: Option<Option<String>>,
    pub(crate) property_changed: bool,
    pub(crate) uniform_changed: bool,
    pub(crate) settings_changed: bool,
    pub(crate) transform_changed: bool,
    pub(crate) dirty: bool,
}

impl<Settings: NamedSettings, Data: DataUniformBuilder + UiDataElement + DataSettings>
    Updater<Settings, Data>
{
    fn new(name: &str) -> Self {
        Self {
            transform: TransformSettings::default(),
            settings: Settings::default().set_name(name),
            data: IndexMap::new(),
            queued_attached_data: Vec::new(),
            attached_data: IndexMap::new(),
            shown_data: None,
            data_to_show: None,
            property_changed: false,
            uniform_changed: false,
            settings_changed: false,
            transform_changed: false,
            dirty: false,
        }
    }

    pub(crate) fn add_data(&mut self, name: String, data: Data) -> &mut Data {
        if let Some(data_name) = &mut self.shown_data {
            if *data_name == name {
                self.data_to_show = Some(Some(data_name.clone()));
                self.shown_data = None;
            }
        }
        let old_data = self.data.insert(name.clone(), data);
        let data = self.data.get_mut(&name).unwrap();
        if let Some(old_data) = old_data {
            data.apply_settings(old_data);
        }
        data
    }

    pub(crate) fn refresh<
        Geometry,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
        Picker: ElementPicker<Geometry = Geometry, Settings = Settings>,
    >(
        &mut self,
        geometry: &Geometry,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        renderer: &mut Renderer<Settings, Data, Geometry, Fixed, DataB, Pipeline>,
        picker: &mut Picker,
        show: bool,
        sbv: &SBV,
        w_sbv: &mut Option<SBV>,
    ) -> bool {
        let mut refresh_screen = self.dirty;
        self.dirty = false;
        if let Some(data) = self.data_to_show.take() {
            self.data_to_show = None;
            self.shown_data = data.clone();
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            renderer.build_data_buffer(device, geometry, data);
            self.property_changed = true;
            refresh_screen = true;
        }
        if self.settings_changed {
            renderer.update_settings(&self.settings, queue);
            picker.update_settings(queue, &self.settings);
            self.settings_changed = false;
            refresh_screen = true;
        }
        if self.transform_changed {
            self.transform
                .to_raw()
                .refresh_buffer(queue, &renderer.transform_uniform);
            self.transform_changed = false;
            picker.update_transform(queue, &self.transform);
            refresh_screen = true;
        }
        if self.property_changed {
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            let data_uniform = data.map(|d| d.build_uniform(device));
            if let Some(data_uniform) = data_uniform {
                renderer.set_data_uniform(data_uniform);
            }
            renderer.build_pipeline(
                device,
                data,
                &self.settings,
                camera_light_bind_group_layout,
                color_format,
            );
            self.property_changed = false;
            self.settings_changed = false;
            self.uniform_changed = false;
            refresh_screen = true;
        } else if self.uniform_changed {
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            if let Some(data) = data {
                if let Some(data_uniform) = renderer.get_data_uniform() {
                    data.refresh_buffer(queue, data_uniform);
                    refresh_screen = true;
                }
            }
            self.uniform_changed = false;
        }
        refresh_screen |= !self.queued_attached_data.is_empty();
        for queued in self.queued_attached_data.drain(..) {
            //TODO recover settings
            let name = queued.name.clone();
            let field = VectorField::new(
                device,
                camera_light_bind_group_layout,
                &renderer.transform_uniform.bind_group_layout,
                color_format,
                queued,
            );
            //let vector_field = VectorFieldData { field, shown };
            self.attached_data.insert(name, field);
        }
        for (_, attached) in &mut self.attached_data {
            refresh_screen |= attached.update(queue);
        }
        if show {
            SBV::merge(w_sbv, &sbv.transform(&self.transform.get_transform()));
        }
        refresh_screen
    }

    fn render_attached_data<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        for (_, attached) in &self.attached_data {
            attached.render(render_pass);
        }
    }

    fn draw(&mut self, ui: &mut egui::Ui, positions: &[[f32; 3]]) {
        self.transform_changed |= self.transform.draw_transform(ui, positions);
        self.settings_changed |= self.settings.draw(ui, &mut self.property_changed);
        for (name, data) in &mut self.data {
            let active = self.shown_data == Some(name.clone());
            egui::CollapsingHeader::new(name)
                .default_open(false)
                .show(ui, |ui| {
                    let mut change_active = active;
                    ui.checkbox(&mut change_active, "Show");
                    if change_active != active {
                        if !active {
                            self.data_to_show = Some(Some(name.clone()))
                        } else {
                            self.data_to_show = Some(None)
                        }
                    }
                    self.uniform_changed |= data.draw(ui, &mut self.property_changed) && active;
                });
        }
        for (name, field) in &mut self.attached_data {
            egui::CollapsingHeader::new(name)
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.checkbox(&mut field.settings.show, "Show").changed() {
                            self.dirty = true;
                        }
                    });
                    //TODO move this
                    if egui::Slider::new(&mut field.settings.magnitude, 0.1..=100.0)
                        .text("Magnitude")
                        .clamping(SliderClamping::Never)
                        .logarithmic(true)
                        .ui(ui)
                        .changed()
                    {
                        field.settings_changed = true;
                    }

                    field.settings_changed |= field.settings.color.draw(ui, &mut false);
                });
        }
    }

    fn draw_gizmo(
        &mut self,
        ui: &mut egui::Ui,
        name: &str,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
        gizmo_hovered: &mut bool,
    ) {
        self.transform_changed |= self
            .transform
            .draw_gizmo(ui, name, view, proj, gizmo_hovered);
        self.settings_changed |= self
            .settings
            .draw_gizmo(ui, name, view, proj, gizmo_hovered);
    }
}

pub(crate) struct Renderer<
    Settings,
    Data,
    Geometry,
    Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
    DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
> {
    pub(crate) fixed: Fixed,
    pub(crate) data_buffer: DataB,
    pub(crate) pipeline: Pipeline,
    pub(crate) transform_uniform: DataUniform,
    pub(crate) settings_uniform: DataUniform,
    pub(crate) data_uniform: Option<DataUniform>,
}

impl<
        Settings: DataUniformBuilder,
        Data,
        Geometry,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
    > Renderer<Settings, Data, Geometry, Fixed, DataB, Pipeline>
{
    pub fn new(
        device: &wgpu::Device,
        geometry: &Geometry,
        transform: &TransformSettings,
        settings: &Settings,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let fixed = Fixed::initialize(device, geometry);
        let data_buffer = DataB::new(device, geometry, None);

        let transform_uniform = transform.to_raw().build_uniform(device).unwrap();
        let settings_uniform = settings.build_uniform(device).unwrap();
        let pipeline = RenderPipeline::new(
            device,
            None,
            &fixed,
            settings,
            &transform_uniform,
            &settings_uniform,
            None,
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
            data_uniform: None,
        }
    }

    fn set_data_uniform(&mut self, data_uniform: Option<DataUniform>) {
        self.data_uniform = data_uniform;
    }

    fn get_data_uniform(&mut self) -> Option<&DataUniform> {
        self.data_uniform.as_ref()
    }

    fn update_settings(&mut self, settings: &Settings, queue: &mut wgpu::Queue) {
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

    fn update_transform(&self, queue: &mut wgpu::Queue, transform: &TransformSettings);

    fn update_settings(&self, queue: &mut wgpu::Queue, settings: &Self::Settings);

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

pub trait Positions {
    fn get_positions(&self) -> &[[f32; 3]];
}
