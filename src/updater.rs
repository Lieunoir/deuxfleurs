use crate::data::{DataUniform, DataUniformBuilder};
use indexmap::IndexMap;

pub struct MainDisplayItem<
    Settings: Default + DataUniformBuilder,
    Data: DataUniformBuilder,
    Item,
    Fixed: FixedRenderer<Settings = Settings, Data = Data, Item = Item>,
    DataB: DataBuffer<Settings = Settings, Data = Data, Item = Item>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Item = Item, Fixed = Fixed>,
> where
    Renderer<Settings, Data, Item, Fixed, DataB, Pipeline>: Render,
{
    pub(crate) item: Item,
    pub(crate) show: bool,
    pub(crate) updater: Updater<Settings, Data>,
    pub(crate) renderer: Renderer<Settings, Data, Item, Fixed, DataB, Pipeline>,
}

impl<
        Settings: Default + DataUniformBuilder,
        Data: DataUniformBuilder,
        Item,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Item = Item>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Item = Item>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Item = Item, Fixed = Fixed>,
    > Render for MainDisplayItem<Settings, Data, Item, Fixed, DataB, Pipeline>
where
    Renderer<Settings, Data, Item, Fixed, DataB, Pipeline>: Render,
{
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show {
            self.renderer.render(render_pass)
        }
    }
}

impl<
        Settings: Default + DataUniformBuilder,
        Data: DataUniformBuilder,
        Item,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Item = Item>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Item = Item>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Item = Item, Fixed = Fixed>,
    > MainDisplayItem<Settings, Data, Item, Fixed, DataB, Pipeline>
where
    Renderer<Settings, Data, Item, Fixed, DataB, Pipeline>: Render,
{
    pub fn init(
        device: &wgpu::Device,
        item: Item,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        //transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let updater = Updater::new();
        let renderer = Renderer::new(
            device,
            &item,
            &updater.settings,
            camera_light_bind_group_layout,
            color_format,
        );
        Self {
            item,
            show: true,
            renderer,
            updater,
        }
    }

    pub fn refresh(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        //renderer: &mut Renderer<Settings, Data, Item, Fixed, DataB, Pipeline>
    ) -> bool {
        self.updater.refresh(
            &self.item,
            device,
            queue,
            camera_light_bind_group_layout,
            color_format,
            &mut self.renderer,
        )
    }
}

pub(crate) struct Updater<Settings: Default, Data: DataUniformBuilder> {
    pub(crate) settings: Settings,
    pub(crate) data: IndexMap<String, Data>,
    pub(crate) shown_data: Option<String>,
    pub(crate) data_to_show: Option<Option<String>>,
    pub(crate) property_changed: bool,
    pub(crate) uniform_changed: bool,
    pub(crate) settings_changed: bool,
}

impl<Settings: Default + DataUniformBuilder, Data: DataUniformBuilder> Updater<Settings, Data> {
    pub fn new() -> Self {
        Self {
            settings: Settings::default(),
            data: IndexMap::new(),
            shown_data: None,
            data_to_show: None,
            property_changed: false,
            uniform_changed: false,
            settings_changed: false,
        }
    }

    pub fn refresh<
        Item,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Item = Item>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Item = Item>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Item = Item, Fixed = Fixed>,
    >(
        &mut self,
        item: &Item,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        renderer: &mut Renderer<Settings, Data, Item, Fixed, DataB, Pipeline>,
    ) -> bool {
        let mut refresh_screen = false;
        if let Some(data) = self.data_to_show.take() {
            self.data_to_show = None;
            self.shown_data = data.clone();
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            renderer.build_data_buffer(device, item, data);
            self.property_changed = true;
            refresh_screen = true;
        }
        if self.settings_changed {
            renderer.update_settings(&self.settings, queue);
            self.settings_changed = false;
            refresh_screen = true;
        }
        if self.property_changed {
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            let data_uniform = data.map(|d| d.build_uniform(device));
            if let Some(data_uniform) = data_uniform {
                renderer.set_data_uniform(data_uniform);
            }
            renderer.build_pipeline(device, data, camera_light_bind_group_layout, color_format);
            self.property_changed = false;
            self.settings_changed = false;
            self.uniform_changed = false;
            refresh_screen = true;
        } else if self.uniform_changed {
            let data_uniform = renderer.get_data_uniform();
            let data = self.shown_data.as_ref().map(|d| self.data.get(d)).flatten();
            if let Some(data) = data {
                if let Some(data_uniform) = renderer.get_data_uniform() {
                    data.refresh_buffer(queue, data_uniform);
                }
            }
        }
        refresh_screen
    }
}

pub(crate) struct Renderer<
    Settings,
    Data,
    Item,
    Fixed: FixedRenderer<Settings = Settings, Data = Data, Item = Item>,
    DataB: DataBuffer<Settings = Settings, Data = Data, Item = Item>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Item = Item, Fixed = Fixed>,
> {
    pub(crate) fixed: Fixed,
    pub(crate) data_buffer: DataB,
    pub(crate) pipeline: Pipeline,
    pub(crate) settings_uniform: DataUniform,
    pub(crate) data_uniform: Option<DataUniform>,
}

impl<
        Settings: DataUniformBuilder,
        Data,
        Item,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Item = Item>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Item = Item>,
        Pipeline: RenderPipeline<Settings = Settings, Data = Data, Item = Item, Fixed = Fixed>,
    > Renderer<Settings, Data, Item, Fixed, DataB, Pipeline>
{
    pub fn new(
        device: &wgpu::Device,
        item: &Item,
        settings: &Settings,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let fixed = Fixed::initialize(device, item, settings);
        let data_buffer = DataB::new(device, item, None);

        let settings_uniform = settings.build_uniform(device).unwrap();
        let pipeline = RenderPipeline::new(
            device,
            None,
            &fixed,
            &settings_uniform,
            None,
            camera_light_bind_group_layout,
            color_format,
        );
        Self {
            fixed,
            data_buffer,
            pipeline,
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

    fn build_data_buffer(&mut self, device: &wgpu::Device, item: &Item, data: Option<&Data>) {
        self.data_buffer = DataB::new(device, item, data);
    }

    fn build_pipeline(
        &mut self,
        device: &wgpu::Device,
        data: Option<&Data>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) {
        self.pipeline = RenderPipeline::new(
            device,
            data,
            &self.fixed,
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
    type Item;

    fn initialize(device: &wgpu::Device, item: &Self::Item, settings: &Self::Settings) -> Self;
}

pub trait DataBuffer {
    type Settings;
    type Data;
    type Item;

    fn new(device: &wgpu::Device, item: &Self::Item, data: Option<&Self::Data>) -> Self;
}

pub trait RenderPipeline {
    type Settings;
    type Data;
    type Item;
    type Fixed;

    fn new(
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        fixed: &Self::Fixed,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self;
}

pub trait Render {
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b;
}
