use crate::attachment::Attachment;
use crate::data::internal::DataSettings;
use crate::data::internal::DataUniform;
use crate::data::internal::DataUniformBuilder;
use crate::data::*;
use crate::shape::AttachedRenderer;
use crate::shape::DataBuffer;
use crate::shape::DataMut;
use crate::shape::FixedRenderer;
use crate::shape::RenderAttached;
use crate::shape::RenderPipeline;
use crate::shape::ShapeDescriptor;
use crate::shape::ShapeGeometry;
use crate::shape::ShapeSettings;
use crate::texture;
use crate::util;
use crate::util::Vertex;
use crate::window::ContextHolder;
use egui::Widget;
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;

pub struct VectorFieldDescriptor;

pub struct VFFixedBuffer {
    vertex_buffer: wgpu::Buffer,
    vector_buffer: wgpu::Buffer,
    vectors_len: u32,
}

pub struct VFPipeline {
    render_pipeline: wgpu::RenderPipeline,
}

#[derive(Clone)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct VFGeometry {
    vectors: Vec<[f32; 3]>,
    offsets: Vec<[f32; 3]>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VectorFieldSettingsRaw {
    magnitude: f32,
    l: f32,
    _padding: [u32; 2],
    color: ColorSettings,
}

#[derive(Clone)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct VectorFieldSettings {
    pub show: bool,
    pub magnitude: f32,
    l: f32,
    pub color: ColorSettings,
}

impl ShapeDescriptor for VectorFieldDescriptor {
    type Data = ();
    type DataBuffer = ();
    type FixedBuffer = VFFixedBuffer;
    type Pipeline = VFPipeline;
    type Geometry = VFGeometry;
    type Settings = VectorFieldSettings;
    type Attached<S: ContextHolder> = ();
}

pub type VectorField<S> = Attachment<S, VectorFieldDescriptor>;

impl VectorFieldSettings {
    fn new(name: &str, l: f32) -> Self {
        Self {
            show: true,
            magnitude: 1.,
            l,
            color: ColorSettings::new(&name),
        }
    }

    fn to_raw(&self) -> VectorFieldSettingsRaw {
        VectorFieldSettingsRaw {
            magnitude: self.magnitude,
            l: self.l,
            _padding: [0; 2],
            color: self.color,
        }
    }
}

impl DataUniformBuilder for VectorFieldSettings {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        self.to_raw().build_uniform(device)
    }

    fn refresh_buffer(&self, queue: &wgpu::Queue, data_uniform: &DataUniform) {
        self.to_raw().refresh_buffer(queue, data_uniform);
    }
}

impl ShapeSettings for VectorFieldSettings {
    fn new(name: &str, characteristic_length: f32) -> Self {
        VectorFieldSettings::new(name, characteristic_length)
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui, _rebuild_pipeline: &mut bool) -> bool {
        let mut settings_changed = false;
        ui.horizontal(|ui| {
            settings_changed |= self.color.draw_ui(ui);
            settings_changed |= egui::DragValue::new(&mut self.magnitude)
                .prefix("Magnitude: ")
                .speed(0.1)
                .ui(ui)
                .changed();
        });
        settings_changed
    }
}

impl FixedRenderer for VFFixedBuffer {
    type Geometry = VFGeometry;

    fn initialize(device: &wgpu::Device, geometry: &Self::Geometry) -> Self {
        let Self::Geometry { vectors, offsets } = geometry;
        let positions = [
            [-0.1, 0., -0.1],
            [0.1, 0., -0.1],
            [-0.1, 0., 0.1],
            [0.1, 0., 0.1],
            [-0.1, 1., 0.1],
            [0.1, 1., 0.1],
            [-0.1, 1., -0.1],
            [0.1, 1., -0.1],
        ];
        let vertices = positions.map(|position| VectorVertex { position });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let mut gpu_vertices = Vec::with_capacity(vectors.len());
        for (vector, offset) in vectors.iter().zip(offsets) {
            let vertex = VectorData {
                orig_position: *offset,
                vector: *vector,
            };
            gpu_vertices.push(vertex);
        }
        let vector_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector Data Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            vertex_buffer,
            vector_buffer,
            vectors_len: vectors.len() as u32,
        }
    }

    fn update_vertex(&mut self, _queue: &wgpu::Queue, _vertex: u32, _geometry: &Self::Geometry) {}
}

impl DataSettings for () {
    fn apply_previous_settings(&mut self, _previous: Self) {}

    fn draw_ui(&mut self, _ui: &mut egui::Ui) -> bool {
        false
    }
}

impl DataBuffer for () {
    type Data = ();
    type Geometry = VFGeometry;

    fn new(_device: &wgpu::Device, _geometry: &Self::Geometry, _data: Option<&Self::Data>) -> Self {
        ()
    }
}

impl RenderPipeline for VFPipeline {
    type Data = ();
    type Geometry = VFGeometry;
    type Settings = VectorFieldSettings;

    fn new(
        device: &wgpu::Device,
        _data: Option<&Self::Data>,
        _geometry: &Self::Geometry,
        _settings: &Self::Settings,
        transform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        _data_uniform: Option<&DataUniform>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        _counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vector Render Pipeline Layout"),
            bind_group_layouts: &[
                Some(camera_bind_group_layout),
                Some(&transform_uniform.bind_group_layout),
                Some(&settings_uniform.bind_group_layout),
            ],
            immediate_size: 0,
        });
        let shader = include_wgsl!("vector_shader.wgsl");
        let render_pipeline = util::create_quad_pipeline(
            device,
            &pipeline_layout,
            Some(texture::DEPTH_FORMAT),
            &[VectorVertex::desc(), VectorData::desc()],
            shader,
            Some("vector field render"),
        );
        Self { render_pipeline }
    }

    fn rebuild(
        &mut self,
        _device: &wgpu::Device,
        _data: Option<&Self::Data>,
        _settings: &Self::Settings,
        _transform_uniform: &DataUniform,
        _settings_uniform: &DataUniform,
        _data_uniform: Option<&DataUniform>,
        _camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
    }
}

impl ShapeGeometry for VFGeometry {
    type Args = (Vec<[f32; 3]>, Vec<[f32; 3]>);
    fn can_be_replaced_by(&self, _other: &Self) -> bool {
        false
    }

    fn get_characteristic_length(&self) -> f32 {
        let avg_vec_length = self.vectors.iter().fold(0., |l, vec| {
            l + (vec[0].powi(0) + vec[1].powi(1) + vec[2].powi(2)).sqrt()
                / self.vectors.len() as f32
        });
        avg_vec_length
    }

    fn get_positions(&self) -> &[[f32; 3]] {
        &self.offsets
    }

    fn get_total_elements(&self) -> u32 {
        self.offsets.len() as u32
    }

    fn get_vertex_pos(&self, vertex: u32) -> [f32; 3] {
        self.offsets[vertex as usize]
    }

    fn move_vertex(
        &mut self,
        vertex: u32,
        pos: [f32; 3],
    ) -> ((Vec<u32>, Vec<[f32; 3]>), (Vec<u32>, Vec<[f32; 3]>)) {
        self.offsets[vertex as usize] = pos;
        ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()))
    }

    fn new(args: Self::Args) -> Self {
        let mut vectors = args.0;
        let offsets = args.1;
        let max_vec_length = vectors.iter().fold(0., |l: f32, vec| {
            l.max((vec[0].powi(0) + vec[1].powi(1) + vec[2].powi(2)).sqrt())
        });
        for v in &mut vectors {
            for c in v {
                *c = *c * 2. / max_vec_length;
            }
        }
        Self { vectors, offsets }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorVertex {
    pub position: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorData {
    pub orig_position: [f32; 3],
    pub vector: [f32; 3],
}

impl Vertex for VectorVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

impl Vertex for VectorData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

impl RenderAttached for AttachedRenderer<VectorFieldDescriptor> {
    fn render_attached<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(2, &self.settings_uniform.bind_group, &[]);
        render_pass.set_pipeline(&self.pipeline.render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.fixed.vector_buffer.slice(..));
        //render_pass.draw(0..18, 0..(self.vectors.len() as u32));
        render_pass.draw(0..8, 0..self.fixed.vectors_len);
    }
}

pub type VectorFieldSettingsMut<'a, Ctxt> = DataMut<'a, &'a mut VectorFieldSettings, Ctxt>;

impl<S: ContextHolder> VectorFieldSettingsMut<'_, S> {
    pub fn set_magnitude(&mut self, magnitude: f32, relative: bool) {
        if relative {
            self.inner.magnitude = magnitude;
        } else {
            self.inner.magnitude = magnitude / self.inner.l;
        }
        self.update_data_settings();
    }

    pub fn set_color(&mut self, color: [f32; 4]) {
        self.inner.color.color = color;
        self.update_data_settings();
    }

    pub fn show(&mut self, show: bool) {
        self.inner.show = show;
        self.update_data_settings();
    }
}
