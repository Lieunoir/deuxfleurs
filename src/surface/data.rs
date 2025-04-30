use crate::data::DataUniform;
use crate::data::DataUniformBuilder;
use crate::data::{ColorMap, ColorMapValues, DataSettings, IsolineSettings, UVMapSettings};
use crate::ui::UiDataElement;
use crate::SurfaceIndices;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexScalarSettingsBuffer {
    isoline: IsolineSettings,
    colormap: ColorMapValues,
}

#[derive(Clone)]
pub struct VertexScalarSettings {
    pub isoline: IsolineSettings,
    pub colormap: ColorMap,
}

impl VertexScalarSettings {
    pub(crate) fn new(values: &[f32]) -> Self {
        Self {
            colormap: ColorMap::new(values),
            isoline: IsolineSettings::default(),
        }
    }

    pub(crate) fn recycle(&mut self, old: Self) {
        self.isoline = old.isoline;
        self.colormap.recycle(old.colormap);
    }

    pub fn set_isolines(&mut self, isolines: f32) {
        self.isoline.isoline_number = isolines;
    }
}

#[derive(Clone)]
pub struct FaceScalarSettings {
    pub colormap: ColorMap,
}

impl FaceScalarSettings {
    pub(crate) fn new(values: &[f32]) -> Self {
        Self {
            colormap: ColorMap::new(values),
        }
    }

    pub(crate) fn recycle(&mut self, old: Self) {
        self.colormap.recycle(old.colormap);
    }
}

impl DataUniformBuilder for FaceScalarSettings {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        self.colormap.get_value().build_uniform(device)
    }

    fn refresh_buffer(&self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        self.colormap
            .get_value()
            .refresh_buffer(queue, data_uniform);
    }
}

impl From<&VertexScalarSettings> for VertexScalarSettingsBuffer {
    fn from(settings: &VertexScalarSettings) -> VertexScalarSettingsBuffer {
        VertexScalarSettingsBuffer {
            isoline: settings.isoline,
            colormap: settings.colormap.get_value(),
        }
    }
}

impl DataUniformBuilder for VertexScalarSettings {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        let settings_buffer: VertexScalarSettingsBuffer = self.into();
        settings_buffer.build_uniform(device)
    }

    fn refresh_buffer(&self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        let settings_buffer: VertexScalarSettingsBuffer = self.into();
        settings_buffer.refresh_buffer(queue, data_uniform);
    }
}

pub enum SurfaceData {
    Color(Vec<[f32; 3]>),
    FaceScalar(Vec<f32>, FaceScalarSettings),
    VertexScalar(Vec<f32>, VertexScalarSettings),
    //TODO think about edge ordering
    //EdgeScalar(Vec<f32>),
    UVMap(Vec<[f32; 2]>, UVMapSettings),
    UVCornerMap(Vec<[f32; 2]>, UVMapSettings),
}

impl DataSettings for SurfaceData {
    fn apply_settings(&mut self, other: Self) {
        match (self, other) {
            (SurfaceData::FaceScalar(_, set1), SurfaceData::FaceScalar(_, set2)) => {
                set1.recycle(set2)
            }
            (SurfaceData::VertexScalar(_, set1), SurfaceData::VertexScalar(_, set2)) => {
                set1.recycle(set2)
            }
            (SurfaceData::UVMap(_, set1), SurfaceData::UVMap(_, set2)) => *set1 = set2,
            (SurfaceData::UVCornerMap(_, set1), SurfaceData::UVCornerMap(_, set2)) => *set1 = set2,
            _ => (),
        }
    }
}

impl DataUniformBuilder for SurfaceData {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        match self {
            SurfaceData::VertexScalar(_, uniform) => uniform.build_uniform(device),
            SurfaceData::FaceScalar(_, uniform) => uniform.build_uniform(device),
            SurfaceData::UVMap(_, uniform) => uniform.build_uniform(device),
            SurfaceData::UVCornerMap(_, uniform) => uniform.build_uniform(device),
            // Maybe use empty uniform instead of none?
            _ => None,
        }
    }

    fn refresh_buffer(&self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        match self {
            SurfaceData::VertexScalar(_, uniform) => uniform.refresh_buffer(queue, data_uniform),
            SurfaceData::FaceScalar(_, uniform) => uniform.refresh_buffer(queue, data_uniform),
            SurfaceData::UVMap(_, uniform) => uniform.refresh_buffer(queue, data_uniform),
            SurfaceData::UVCornerMap(_, uniform) => uniform.refresh_buffer(queue, data_uniform),
            _ => (),
        }
    }
}

impl UiDataElement for SurfaceData {
    fn draw(&mut self, ui: &mut egui::Ui, property_changed: &mut bool) -> bool {
        match self {
            SurfaceData::UVMap(_, data_uniform) | SurfaceData::UVCornerMap(_, data_uniform) => {
                data_uniform.draw(ui, property_changed)
            }
            SurfaceData::VertexScalar(_, data_uniform) => {
                let changed = data_uniform.colormap.draw(ui, property_changed);
                data_uniform.isoline.draw(ui, property_changed) || changed
            }
            SurfaceData::FaceScalar(_, data_uniform) => {
                data_uniform.colormap.draw(ui, property_changed)
            }
            _ => false,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexColorData {
    color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexScalarData {
    scalar: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexUVData {
    uv: [f32; 2],
}

use crate::util::Vertex;

impl Vertex for VertexColorData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

impl Vertex for VertexScalarData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32,
            }],
        }
    }
}

impl Vertex for VertexUVData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}

impl SurfaceData {
    pub(crate) fn desc<'a>(&self) -> wgpu::VertexBufferLayout<'a> {
        match self {
            SurfaceData::Color(..) => VertexColorData::desc(),
            SurfaceData::FaceScalar(..) | SurfaceData::VertexScalar(..) => VertexScalarData::desc(),
            SurfaceData::UVMap(..) | SurfaceData::UVCornerMap(..) => VertexUVData::desc(),
        }
    }

    pub(crate) fn build_vertex_buffer(
        &self,
        device: &wgpu::Device,
        indices: &SurfaceIndices,
        internal_indices: &[[u32; 3]],
    ) -> wgpu::Buffer {
        match self {
            SurfaceData::Color(colors) => {
                let mut gpu_vertices = Vec::with_capacity(3 * internal_indices.len());
                for face in internal_indices {
                    for index in face {
                        gpu_vertices.push(VertexColorData {
                            color: colors[*index as usize],
                        });
                    }
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Data Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            SurfaceData::VertexScalar(datas, _) => {
                let mut min_d = datas[0];
                let mut max_d = datas[0];
                for data in datas {
                    if *data > max_d {
                        max_d = *data;
                    }
                    if *data < min_d {
                        min_d = *data;
                    }
                }

                let mut gpu_vertices = Vec::with_capacity(3 * internal_indices.len());
                for face in internal_indices {
                    for index in face {
                        let data = datas[*index as usize];
                        let t = (data - min_d) / (max_d - min_d);
                        gpu_vertices.push(VertexScalarData { scalar: t });
                    }
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Data Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            SurfaceData::FaceScalar(datas, _) => {
                let mut min_d = datas[0];
                let mut max_d = datas[0];
                for data in datas {
                    if *data > max_d {
                        max_d = *data;
                    }
                    if *data < min_d {
                        min_d = *data;
                    }
                }
                let mut gpu_vertices = Vec::with_capacity(3 * internal_indices.len());
                for (face, data) in indices.into_iter().zip(datas) {
                    let t = (data - min_d) / (max_d - min_d);
                    for _i in 1..face.len() - 1 {
                        gpu_vertices.push(VertexScalarData { scalar: t });
                        gpu_vertices.push(VertexScalarData { scalar: t });
                        gpu_vertices.push(VertexScalarData { scalar: t });
                    }
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Data Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            SurfaceData::UVMap(uv_map, _) => {
                let mut gpu_vertices = Vec::with_capacity(3 * internal_indices.len());
                for face in internal_indices {
                    for index in face {
                        gpu_vertices.push(VertexUVData {
                            uv: uv_map[*index as usize],
                        });
                    }
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Data Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            SurfaceData::UVCornerMap(uv_map, _) => {
                //TODO this but for polygonal faces
                let gpu_vertices: Vec<_> =
                    uv_map.iter().map(|uv| VertexUVData { uv: *uv }).collect();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Data Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
        }
    }
}
