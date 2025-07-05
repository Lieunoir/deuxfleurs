use crate::Settings;
use crate::data::Colors;
use crate::data::{internal::*, *};
use crate::shape::Context;
use crate::shape::DataMut;
use crate::shape::DataMutTrait;
use crate::types::SurfaceIndices;
use crate::ui::UiDataElement;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexScalarSettingsBuffer {
    isoline: IsolineSettings,
    colormap: ColorMapValues,
}

#[derive(Clone)]
pub struct VertexScalarSettings {
    isoline: IsolineSettings,
    colormap: ColorMap,
}

impl VertexScalarSettings {
    pub(crate) fn new(values: &[f32], settings: &Settings) -> Self {
        Self {
            colormap: ColorMap::new(values, settings),
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

pub type VertexScalarSettingsMut<'a, Ctxt> = DataMut<'a, &'a mut VertexScalarSettings, Ctxt>;

impl<'a, Ctxt: Context> VertexScalarSettingsMut<'a, Ctxt>
where
    Self: DataMutTrait,
{
    pub fn set_isolines(&mut self, number: f32) {
        self.inner.isoline.isoline_number = number;
        self.update_data_settings();
    }

    pub fn set_colormap(&mut self, colormap: Colors) {
        self.inner.colormap.colors = colormap;
        self.update_data_settings();
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

    fn refresh_buffer(&self, queue: &wgpu::Queue, data_uniform: &DataUniform) {
        let settings_buffer: VertexScalarSettingsBuffer = self.into();
        settings_buffer.refresh_buffer(queue, data_uniform);
    }
}

pub enum SurfaceData {
    Color(Vec<[f32; 3]>),
    FaceScalar(Vec<f32>, ColorMap),
    VertexScalar(Vec<f32>, VertexScalarSettings),
    EdgeScalar(Vec<f32>, ColorMap),
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
            (SurfaceData::EdgeScalar(_, set1), SurfaceData::EdgeScalar(_, set2)) => {
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
            SurfaceData::FaceScalar(_, uniform) => uniform.get_value().build_uniform(device),
            SurfaceData::EdgeScalar(_, uniform) => uniform.get_value().build_uniform(device),
            SurfaceData::UVMap(_, uniform) => uniform.build_uniform(device),
            SurfaceData::UVCornerMap(_, uniform) => uniform.build_uniform(device),
            // Maybe use empty uniform instead of none?
            _ => None,
        }
    }

    fn refresh_buffer(&self, queue: &wgpu::Queue, data_uniform: &DataUniform) {
        match self {
            SurfaceData::VertexScalar(_, uniform) => uniform.refresh_buffer(queue, data_uniform),
            SurfaceData::FaceScalar(_, uniform) => {
                uniform.get_value().refresh_buffer(queue, data_uniform)
            }
            SurfaceData::EdgeScalar(_, uniform) => {
                uniform.get_value().refresh_buffer(queue, data_uniform)
            }
            SurfaceData::UVMap(_, uniform) => uniform.refresh_buffer(queue, data_uniform),
            SurfaceData::UVCornerMap(_, uniform) => uniform.refresh_buffer(queue, data_uniform),
            _ => (),
        }
    }
}

impl UiDataElement for SurfaceData {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        match self {
            SurfaceData::UVMap(_, data_uniform) | SurfaceData::UVCornerMap(_, data_uniform) => {
                data_uniform.draw_ui(ui)
            }
            SurfaceData::VertexScalar(_, data_uniform) => {
                let mut changed = false;
                ui.horizontal_wrapped(|ui| {
                    changed |= data_uniform.isoline.draw_ui(ui);
                    changed |= data_uniform.colormap.draw_ui(ui);
                });
                changed
            }
            SurfaceData::FaceScalar(_, data_uniform) | SurfaceData::EdgeScalar(_, data_uniform) => {
                data_uniform.draw_ui(ui)
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
struct VertexTripleScalarData {
    scalar: [f32; 3],
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

impl Vertex for VertexTripleScalarData {
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
            SurfaceData::EdgeScalar(..) => VertexTripleScalarData::desc(),
            SurfaceData::UVMap(..) | SurfaceData::UVCornerMap(..) => VertexUVData::desc(),
        }
    }

    pub(crate) fn build_vertex_buffer(
        &self,
        device: &wgpu::Device,
        indices: &SurfaceIndices,
        face_to_edge: &[u32],
    ) -> wgpu::Buffer {
        match self {
            SurfaceData::Color(colors) => {
                let mut gpu_vertices = Vec::with_capacity(3 * indices.tot_triangles());
                for face in indices {
                    for i in 1..face.len() - 1 {
                        for index in [face[0], face[i], face[i + 1]] {
                            gpu_vertices.push(VertexColorData {
                                color: colors[index as usize],
                            });
                        }
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

                let mut gpu_vertices = Vec::with_capacity(3 * indices.tot_triangles());
                for face in indices {
                    for i in 1..face.len() - 1 {
                        for index in [face[0], face[i], face[i + 1]] {
                            let data = datas[index as usize];
                            let t = (data - min_d) / (max_d - min_d);
                            gpu_vertices.push(VertexScalarData { scalar: t });
                        }
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
                let mut gpu_vertices = Vec::with_capacity(3 * indices.tot_triangles());
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
            SurfaceData::EdgeScalar(datas, _) => {
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
                let mut gpu_vertices = Vec::with_capacity(3 * indices.tot_triangles());
                let mut offset = 0;
                for face in indices.into_iter() {
                    if face.len() == 3 {
                        let data_0 = datas[face_to_edge[offset + 1] as usize];
                        let t_0 = (data_0 - min_d) / (max_d - min_d);
                        let data_1 = datas[face_to_edge[offset + 2] as usize];
                        let t_1 = (data_1 - min_d) / (max_d - min_d);
                        let data_2 = datas[face_to_edge[offset] as usize];
                        let t_2 = (data_2 - min_d) / (max_d - min_d);
                        let values = [t_0, t_1, t_2];
                        gpu_vertices.push(VertexTripleScalarData { scalar: values });
                        gpu_vertices.push(VertexTripleScalarData { scalar: values });
                        gpu_vertices.push(VertexTripleScalarData { scalar: values });
                    } else {
                        for j in 1..(face.len() - 1) {
                            let values = if j == 1 {
                                let data_0 = datas[face_to_edge[offset + 1] as usize];
                                let t_0 = (data_0 - min_d) / (max_d - min_d);
                                let data_2 = datas[face_to_edge[offset] as usize];
                                let t_2 = (data_2 - min_d) / (max_d - min_d);
                                [t_0, 0., t_2]
                            } else if j == face.len() - 2 {
                                let data_0 = datas[face_to_edge[offset + j] as usize];
                                let t_0 = (data_0 - min_d) / (max_d - min_d);
                                let data_1 = datas[face_to_edge[offset + j + 1] as usize];
                                let t_1 = (data_1 - min_d) / (max_d - min_d);
                                [t_0, t_1, 0.]
                            } else {
                                let data_0 = datas[face_to_edge[offset + j] as usize];
                                let t_0 = (data_0 - min_d) / (max_d - min_d);
                                [t_0, 0., 0.]
                            };
                            gpu_vertices.push(VertexTripleScalarData { scalar: values });
                            gpu_vertices.push(VertexTripleScalarData { scalar: values });
                            gpu_vertices.push(VertexTripleScalarData { scalar: values });
                        }
                    }
                    offset += face.len();
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Data Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            SurfaceData::UVMap(uv_map, _) => {
                let mut gpu_vertices = Vec::with_capacity(3 * indices.tot_triangles());
                for face in indices {
                    for i in 1..face.len() - 1 {
                        for index in [face[0], face[i], face[i + 1]] {
                            gpu_vertices.push(VertexUVData {
                                uv: uv_map[index as usize],
                            });
                        }
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
