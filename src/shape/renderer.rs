use crate::data::TransformSettings;
use crate::data::internal::{DataUniform, DataUniformBuilder};
use crate::shape::ShapeDescriptor;

pub struct Renderer<Desc: ShapeDescriptor + ?Sized> {
    // Buffers which won't be modified after init
    // (actually can due to geometry replacement)
    pub(crate) fixed: Desc::FixedBuffer,
    // Buffers modified depending on displayed data
    pub(crate) data_buffer: Desc::DataBuffer,
    // Pipeline, may be modifed depending on data, data settings or shape settings
    pub(crate) pipeline: Desc::Pipeline,
    // Common shape uniforms
    pub(crate) transform_uniform: DataUniform,
    pub(crate) settings_uniform: DataUniform,
    // Uniforms modified depending on displayed data and data settings
    pub(crate) data_uniform: Option<DataUniform>,
}

pub struct AttachedRenderer<Desc: ShapeDescriptor + ?Sized> {
    pub(crate) fixed: Desc::FixedBuffer,
    pub(crate) pipeline: Desc::Pipeline,
    pub(crate) settings_uniform: DataUniform,
}

impl<Desc: ShapeDescriptor + ?Sized> Renderer<Desc> {
    pub(crate) fn new(
        device: &wgpu::Device,
        geometry: &Desc::Geometry,
        transform: &TransformSettings,
        settings: &Desc::Settings,
        data: Option<&Desc::Data>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let fixed = Desc::FixedBuffer::initialize(device, geometry);
        let data_buffer = Desc::DataBuffer::new(device, geometry, data);

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
            camera_bind_group_layout,
            counter_bind_group_layout,
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

    pub(crate) fn get_data_uniform(&mut self) -> Option<&DataUniform> {
        self.data_uniform.as_ref()
    }

    pub(crate) fn rebuild_fixed_buffer(
        &mut self,
        device: &wgpu::Device,
        geometry: &Desc::Geometry,
    ) {
        self.fixed = Desc::FixedBuffer::initialize(device, geometry)
    }

    pub(crate) fn rebuild_data_buffer(
        &mut self,
        device: &wgpu::Device,
        geometry: &Desc::Geometry,
        data: Option<&Desc::Data>,
        settings: &Desc::Settings,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        self.data_buffer = Desc::DataBuffer::new(device, geometry, data);
        self.data_uniform = data.map(|d| d.build_uniform(device)).flatten();
        self.rebuild_pipeline(device, data, settings, camera_bind_group_layout);
    }

    pub(crate) fn rebuild_pipeline(
        &mut self,
        device: &wgpu::Device,
        data: Option<&Desc::Data>,
        settings: &Desc::Settings,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        self.pipeline.rebuild(
            device,
            data,
            settings,
            &self.transform_uniform,
            &self.settings_uniform,
            self.data_uniform.as_ref(),
            camera_bind_group_layout,
        );
    }
}

pub trait FixedRenderer {
    type Geometry;

    fn initialize(device: &wgpu::Device, geometry: &Self::Geometry) -> Self;

    fn update_vertex(&mut self, queue: &wgpu::Queue, vertex: u32, geometry: &Self::Geometry);
}

pub trait DataBuffer {
    type Data;
    type Geometry;

    fn new(device: &wgpu::Device, geometry: &Self::Geometry, data: Option<&Self::Data>) -> Self;
}

pub trait RenderPipeline {
    type Settings;
    type Data;
    type Geometry;

    fn new(
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        geometry: &Self::Geometry,
        settings: &Self::Settings,
        tansform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self;

    fn rebuild(
        &mut self,
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        settings: &Self::Settings,
        transform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    );
}

pub trait Render {
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

impl<Desc: ShapeDescriptor + ?Sized> AttachedRenderer<Desc> {
    pub(crate) fn new(
        device: &wgpu::Device,
        geometry: &Desc::Geometry,
        settings: &Desc::Settings,
        transform_uniform: &DataUniform,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let fixed = Desc::FixedBuffer::initialize(device, geometry);
        let settings_uniform = settings.build_uniform(device).unwrap();
        let pipeline = RenderPipeline::new(
            device,
            None,
            geometry,
            settings,
            &transform_uniform,
            &settings_uniform,
            None,
            camera_bind_group_layout,
            counter_bind_group_layout,
        );
        Self {
            fixed,
            pipeline,
            settings_uniform,
        }
    }
}
