use crate::attachment::internal::AttachmentPosition;
use crate::data::*;
use crate::shape::AttachedGeometry;
use crate::shape::Context;
use crate::shape::{DataMut, DataMutTrait};

use crate::shape::GraphicalContext;
use crate::shape::NewAttachedGeometry;
use crate::texture;
use crate::ui::UiDataElement;
use crate::util;
use crate::util::Vertex;
use egui::Widget;
use serde::Deserialize;
use serde::Serialize;
use wgpu::BufferAddress;
use wgpu::util::DeviceExt;

pub struct VectorField {
    position: AttachmentPosition,
    pub vectors: Vec<[f32; 3]>,
    pub offsets: Vec<[f32; 3]>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vector_buffer: wgpu::Buffer,
    pub settings: VectorFieldSettings,
    settings_bind_group: wgpu::BindGroup,
    settings_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VectorFieldSettingsRaw {
    magnitude: f32,
    l: f32,
    _padding: [u32; 2],
    color: ColorSettings,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct VectorFieldSettings {
    pub show: bool,
    pub magnitude: f32,
    l: f32,
    pub color: ColorSettings,
}

impl VectorFieldSettings {
    fn new(l: f32) -> Self {
        Self {
            show: true,
            magnitude: 1.,
            l,
            color: ColorSettings::default(),
        }
    }
}

impl VectorFieldSettings {
    fn to_raw(&self) -> VectorFieldSettingsRaw {
        VectorFieldSettingsRaw {
            magnitude: self.magnitude,
            l: self.l,
            _padding: [0; 2],
            color: self.color,
        }
    }
}

pub type VectorFieldSettingsMut<'a, Ctxt> = DataMut<'a, &'a mut VectorFieldSettings, Ctxt>;

impl<'a, Ctxt: Context> VectorFieldSettingsMut<'a, Ctxt>
where
    Self: DataMutTrait,
{
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

#[derive(Serialize, Deserialize)]
pub struct NewVectorField {
    position: AttachmentPosition,
    vectors: Vec<[f32; 3]>,
    offsets: Vec<[f32; 3]>,
    pub(crate) settings: VectorFieldSettings,
}

impl NewVectorField {
    pub(crate) fn new(
        name: String,
        characteristic_l: f32,
        position: AttachmentPosition,
        vectors: Vec<[f32; 3]>,
        offsets: Vec<[f32; 3]>,
    ) -> NewVectorField {
        let avg_vec_length = vectors.iter().fold(0., |l, vec| {
            l + (vec[0].powi(0) + vec[1].powi(1) + vec[2].powi(2)).sqrt() / vectors.len() as f32
        });
        let mut settings = VectorFieldSettings::new(characteristic_l / avg_vec_length * 2.);
        settings.color = ColorSettings::new(&name);
        NewVectorField {
            position,
            vectors,
            offsets,
            settings,
        }
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

impl VectorField {
    fn build_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
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
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    fn build_vector_buffer(
        device: &wgpu::Device,
        vectors: &Vec<[f32; 3]>,
        offsets: &Vec<[f32; 3]>,
    ) -> wgpu::Buffer {
        let mut gpu_vertices = Vec::with_capacity(vectors.len());
        for (vector, offset) in vectors.iter().zip(offsets) {
            let vertex = VectorData {
                orig_position: *offset,
                vector: *vector,
            };
            gpu_vertices.push(vertex);
        }
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector Data Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn new(
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        NewVectorField {
            position,
            vectors,
            offsets,
            settings,
        }: NewVectorField,
    ) -> Self {
        assert!(vectors.len() == offsets.len());

        let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector field settings buffer"),
            contents: bytemuck::cast_slice(&[settings.to_raw()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let settings_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("vector_field_settings_bind_group_layout"),
            });
        let settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &settings_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: settings_buffer.as_entire_binding(),
            }],
            label: Some("vector_field_settings_bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vector Render Pipeline Layout"),
            bind_group_layouts: &[
                camera_light_bind_group_layout,
                transform_bind_group_layout,
                &settings_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("arrow shader"),
            source: wgpu::ShaderSource::Wgsl(super::vector_shader::ARROW_SHADER.into()),
        };
        let render_pipeline = util::create_quad_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[VectorVertex::desc(), VectorData::desc()],
            shader,
            Some("vector field render"),
        );

        let vertex_buffer = Self::build_vertex_buffer(device);
        let vector_buffer = Self::build_vector_buffer(device, &vectors, &offsets);
        Self {
            position,
            vectors,
            offsets: offsets,
            render_pipeline,
            vertex_buffer,
            vector_buffer,
            settings,
            settings_bind_group,
            settings_buffer,
        }
    }
}

impl<'a> AttachedGeometry<&'a mut crate::Settings> for NewVectorField {
    type Args = (Vec<[f32; 3]>, Vec<[f32; 3]>);
    type Settings<'b> = &'b mut VectorFieldSettings;

    fn new(
        name: String,
        args: Self::Args,
        position: AttachmentPosition,
        characteristic_l: f32,
        _context: &mut &'a mut crate::Settings,
        _transform_layout: &(),
    ) -> Self {
        let (vectors, offsets) = args;
        NewVectorField::new(name, characteristic_l, position, vectors, offsets)
    }

    fn get_settings(&mut self) -> Self::Settings<'_> {
        &mut self.settings
    }

    fn get_attached_position(&self) -> &AttachmentPosition {
        &self.position
    }

    fn move_elements(&mut self, _queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]) {
        for (index, value) in indices.iter().zip(pos) {
            self.offsets[*index as usize] = *value;
        }
    }
}

impl<'a> AttachedGeometry<GraphicalContext<'a>> for VectorField {
    type Args = (Vec<[f32; 3]>, Vec<[f32; 3]>);
    type Settings<'b> = &'b mut VectorFieldSettings;

    fn new(
        name: String,
        args: Self::Args,
        position: AttachmentPosition,
        characteristic_l: f32,
        context: &mut GraphicalContext<'a>,
        transform_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        *context.refresh_screen = true;
        let (vectors, offsets) = args;
        let new_vector_field =
            NewVectorField::new(name, characteristic_l, position, vectors, offsets);
        VectorField::new(
            context.device,
            context.camera_light_bind_group_layout,
            transform_layout,
            context.color_format,
            new_vector_field,
        )
    }

    fn shown(&self) -> bool {
        self.settings.show
    }

    fn show(&mut self, show: bool, refresh_screen: &mut bool) {
        self.settings.show = show;
        *refresh_screen = true;
    }

    fn draw_ui(
        &mut self,
        ui: &mut egui::Ui,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
        refresh_screen: &mut bool,
    ) {
        let mut settings_changed = false;
        //TODO move this

        ui.horizontal(|ui| {
            settings_changed |= self.settings.color.draw_ui(ui);
            settings_changed |= egui::DragValue::new(&mut self.settings.magnitude)
                .prefix("Magnitude: ")
                .speed(0.1)
                .ui(ui)
                .changed();
        });
        if settings_changed {
            *refresh_screen = true;
            queue.write_buffer(
                &self.settings_buffer,
                0,
                bytemuck::cast_slice(&[self.settings.to_raw()]),
            );
        }
    }

    fn render<'c, 'd>(&'c self, render_pass: &mut wgpu::RenderPass<'d>)
    where
        'c: 'd,
    {
        render_pass.set_bind_group(2, &self.settings_bind_group, &[]);
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.vector_buffer.slice(..));
        //render_pass.draw(0..18, 0..(self.vectors.len() as u32));
        render_pass.draw(0..8, 0..(self.vectors.len() as u32));
    }

    fn get_settings(&mut self) -> Self::Settings<'_> {
        &mut self.settings
    }

    fn get_attached_position(&self) -> &AttachmentPosition {
        &self.position
    }

    fn move_elements(&mut self, queue: &wgpu::Queue, indices: &[u32], pos: &[[f32; 3]]) {
        for (index, value) in indices.iter().zip(pos) {
            self.offsets[*index as usize] = *value;
            queue.write_buffer(
                &self.vector_buffer,
                (*index as usize * size_of::<VectorData>()) as BufferAddress,
                bytemuck::cast_slice(&[VectorData {
                    orig_position: *value,
                    vector: self.vectors[*index as usize],
                }]),
            );
        }
    }
}

impl NewAttachedGeometry for NewVectorField {
    type UpgradedAttachedGeometry = VectorField;

    fn init(
        self,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self::UpgradedAttachedGeometry {
        VectorField::new(
            device,
            camera_light_bind_group_layout,
            transform_bind_group_layout,
            color_format,
            self,
        )
    }

    fn downgrade(upgraded: &Self::UpgradedAttachedGeometry) -> Self {
        Self {
            position: upgraded.position.clone(),
            settings: upgraded.settings.clone(),
            vectors: upgraded.vectors.clone(),
            offsets: upgraded.offsets.clone(),
        }
    }
}
