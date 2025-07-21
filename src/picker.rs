use crate::camera::Camera;

use crate::DisplayPointCloud;
use crate::DisplaySegment;
use crate::DisplaySurface;
use crate::shape::ShapeGeometry;
use crate::texture;
use crate::util;
use crate::util::BufferDimensions;
use crate::window::UserEvent;
use egui::Checkbox;
use indexmap::IndexMap;
use transform_gizmo_egui::GizmoOrientation;
use transform_gizmo_egui::{Gizmo, GizmoConfig, GizmoExt, GizmoMode, enum_set};
use wgpu::util::DeviceExt;
use winit::event::*;

pub(crate) struct Picker {
    pub picked_item: Option<(String, Picked)>,
    item_to_pick: Option<(usize, usize)>,
    //lock to ensure buffer isn't used while mapped
    pub pick_locked: bool,
    dragging: bool,
    cur_pos: (f32, f32),
    orig_pos: (f32, f32),
    pub bind_group_layout: wgpu::BindGroupLayout,
    bind_groups: Vec<wgpu::BindGroup>,
    pub counters_dirty: bool,
    width: u32,
    height: u32,
    gizmo: Gizmo,
    show_gizmo: bool,
}

#[derive(PartialEq, Clone)]
pub enum SurfacePicked {
    Vertex(u32),
    Face(u32),
    Edge(u32),
}

#[derive(PartialEq, Clone)]
pub enum SegmentPicked {
    Point(u32),
    Edge(u32),
}

#[derive(PartialEq, Clone)]
pub enum Picked {
    Surface(SurfacePicked),
    PointCloud(u32),
    Segment(SegmentPicked),
}

impl Picked {}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CounterUniform {
    count: u32,
    _padding_1: u32,
    _padding_2: u32,
    _padding_3: u32,
}

impl Picker {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("picker_counter_bind_group_layout"),
        });
        let bind_groups = Vec::new();

        Self {
            picked_item: None,
            item_to_pick: None,
            pick_locked: false,
            dragging: false,
            cur_pos: (0., 0.),
            orig_pos: (0., 0.),
            bind_group_layout,
            bind_groups,
            counters_dirty: true,
            width,
            height,
            gizmo: Gizmo::new(GizmoConfig::default()),
            show_gizmo: false,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.item_to_pick = None;
        self.width = width;
        self.height = height;
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cur_pos = (position.x as f32, position.y as f32);
                let dx = self.cur_pos.0 - self.orig_pos.0;
                let dy = self.cur_pos.1 - self.orig_pos.1;
                if dx * dx + dy * dy > 5. {
                    self.dragging = true;
                }
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    if *state == ElementState::Pressed {
                        self.dragging = false;
                        self.orig_pos = self.cur_pos;
                        false
                    } else if !self.dragging {
                        self.item_to_pick =
                            Some((self.cur_pos.0 as usize, self.cur_pos.1 as usize));
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture_view: &wgpu::TextureView,
        depth_texture_view: &wgpu::TextureView,
        camera_light_bind_group: &wgpu::BindGroup,
        surfaces: &IndexMap<String, DisplaySurface>,
        clouds: &IndexMap<String, DisplayPointCloud>,
        curves: &IndexMap<String, DisplaySegment>,
    ) -> bool {
        if !self.pick_locked && self.item_to_pick.is_some() {
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Picker Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_texture_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                render_pass.set_bind_group(0, camera_light_bind_group, &[]);

                if self.counters_dirty {
                    let mut counter = 1;
                    self.bind_groups = surfaces
                        .values()
                        .map(|surface| {
                            let counter_uniform = CounterUniform {
                                count: counter,
                                _padding_1: 0,
                                _padding_2: 0,
                                _padding_3: 0,
                            };
                            counter += surface.geometry().get_total_elements();

                            //TODO use one dynamic buffer instead
                            let counter_buffer =
                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("counter buffer"),
                                    contents: bytemuck::cast_slice(&[counter_uniform]),
                                    usage: wgpu::BufferUsages::UNIFORM
                                        | wgpu::BufferUsages::COPY_DST,
                                });
                            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                layout: &self.bind_group_layout,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: counter_buffer.as_entire_binding(),
                                }],
                                label: Some("camera_light_bind_group"),
                            });
                            bind_group
                        })
                        .collect();

                    for cloud in clouds.values() {
                        let counter_uniform = CounterUniform {
                            count: counter,
                            _padding_1: 0,
                            _padding_2: 0,
                            _padding_3: 0,
                        };
                        counter += cloud.geometry().get_total_elements();

                        let counter_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("counter buffer"),
                                contents: bytemuck::cast_slice(&[counter_uniform]),
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            });
                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.bind_group_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: counter_buffer.as_entire_binding(),
                            }],
                            label: Some("camera_light_bind_group"),
                        });
                        self.bind_groups.push(bind_group);
                    }

                    for curve in curves.values() {
                        let counter_uniform = CounterUniform {
                            count: counter,
                            _padding_1: 0,
                            _padding_2: 0,
                            _padding_3: 0,
                        };
                        counter += curve.geometry().get_total_elements();

                        let counter_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("counter buffer"),
                                contents: bytemuck::cast_slice(&[counter_uniform]),
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            });
                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.bind_group_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: counter_buffer.as_entire_binding(),
                            }],
                            label: Some("camera_light_bind_group"),
                        });
                        self.bind_groups.push(bind_group);
                    }

                    self.counters_dirty = false;
                }

                let mut index = 0;
                for surface in surfaces.values() {
                    let counter_bind_group = &self.bind_groups[index];
                    index += 1;
                    render_pass.set_bind_group(1, counter_bind_group, &[]);
                    surface.render_picker(&mut render_pass);
                }
                for cloud in clouds.values() {
                    let counter_bind_group = &self.bind_groups[index];
                    index += 1;
                    render_pass.set_bind_group(1, counter_bind_group, &[]);
                    cloud.render_picker(&mut render_pass);
                }
                for curve in curves.values() {
                    let counter_bind_group = &self.bind_groups[index];
                    index += 1;
                    render_pass.set_bind_group(1, counter_bind_group, &[]);
                    curve.render_picker(&mut render_pass);
                }
            }
            true
        } else {
            false
        }
    }

    pub fn post_render(
        &mut self,
        event_loop_proxy: &winit::event_loop::EventLoopProxy<UserEvent>,
        buffer: &wgpu::Buffer,
        buffer_dimensions: &BufferDimensions,
    ) {
        if let Some((i, j)) = self.item_to_pick
            && !self.pick_locked
        {
            // Use slice more efficiently
            let index = (j * buffer_dimensions.padded_bytes_per_row + 4 * i) as wgpu::BufferAddress;
            let index = index - index % wgpu::MAP_ALIGNMENT;
            let buffer_slice = buffer.slice(index..index + wgpu::MAP_ALIGNMENT);
            let event_loop_proxy = event_loop_proxy.clone();

            #[cfg(not(target_arch = "wasm32"))]
            {
                buffer_slice.map_async(wgpu::MapMode::Read, move |_v| {
                    event_loop_proxy.send_event(UserEvent::Pick).ok();
                });
            }
            #[cfg(target_arch = "wasm32")]
            {
                // `event_loop_proxy` isn't `Send` so we have to find another way
                let (sender, receiver) = oneshot::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |_v| {
                    sender.send(()).unwrap();
                });
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(()) = receiver.await {
                        event_loop_proxy.send_event(UserEvent::Pick).ok();
                    }
                });
            }
            //buffer_slice = Some(buffer_slice_t);
            self.pick_locked = true;
        }
    }

    pub fn pick(
        &mut self,
        surfaces: &IndexMap<String, DisplaySurface>,
        clouds: &IndexMap<String, DisplayPointCloud>,
        curves: &IndexMap<String, DisplaySegment>,
        camera: &Camera,
        buffer: &wgpu::Buffer,
        buffer_dimensions: &BufferDimensions,
    ) {
        {
            if let Some((i, j)) = self.item_to_pick {
                let index =
                    (j * buffer_dimensions.padded_bytes_per_row + 4 * i) as wgpu::BufferAddress;
                let aligned_index = index - index % wgpu::MAP_ALIGNMENT;
                let buffer_slice = buffer.slice(aligned_index..aligned_index + wgpu::MAP_ALIGNMENT);
                let data_offset = (index - aligned_index) as usize;
                let data = buffer_slice.get_mapped_range();
                let value = (data[data_offset + 3] as u32) << 24
                    | (data[data_offset + 2] as u32) << 16
                    | (data[data_offset + 1] as u32) << 8
                    | (data[data_offset + 0] as u32);
                let mut c = 1;
                if let Some((name, picked)) = surfaces
                    .iter()
                    .find(|(_key, surface)| {
                        let found =
                            c <= value && value < c + surface.geometry().get_total_elements();
                        if !found {
                            c += surface.geometry().get_total_elements();
                        }
                        found
                    })
                    .map(|(n, s)| {
                        let pos_x = (i as f32 / self.width as f32) * 2. - 1.;
                        let pos_y = -((j as f32 / self.height as f32) * 2. - 1.);
                        (
                            n,
                            Picked::Surface(s.get_element(camera, value - c, pos_x, pos_y)),
                        )
                    })
                    .or_else(|| {
                        clouds
                            .iter()
                            .find(|(_key, cloud)| {
                                let found =
                                    c <= value && value < c + cloud.geometry().get_total_elements();
                                if !found {
                                    c += cloud.geometry().get_total_elements();
                                }
                                found
                            })
                            .map(|(n, _pc)| (n, Picked::PointCloud(value - c)))
                    })
                    .or_else(|| {
                        curves
                            .iter()
                            .find(|(_key, curve)| {
                                let found =
                                    c <= value && value < c + curve.geometry().get_total_elements();
                                if !found {
                                    c += curve.geometry().get_total_elements();
                                }
                                found
                            })
                            .map(|(n, curve)| (n, Picked::Segment(curve.get_element(value - c))))
                    })
                {
                    self.picked_item = Some((name.clone(), picked));
                } else {
                    self.picked_item = None;
                }
                self.item_to_pick = None;
            }
        }
        buffer.unmap();
        self.pick_locked = false;
    }

    pub(crate) fn draw_ui(&mut self, ui: &mut egui::Ui) {
        if let Some((picked_name, picked)) = self.picked_item.as_ref() {
            ui.separator();
            ui.heading("Selection");
            ui.label(format!("Shape: {}", picked_name));

            match picked {
                Picked::Surface(picked) => match picked {
                    SurfacePicked::Vertex(picked) => {
                        ui.label(format!("Vertex number {}", picked));
                    }
                    SurfacePicked::Face(picked) => {
                        ui.label(format!("Face number {}", picked));
                    }
                    SurfacePicked::Edge(picked) => {
                        ui.label(format!("Edge number {}", picked));
                    }
                },
                Picked::PointCloud(picked) => {
                    ui.label(format!("Point number {}", picked));
                }
                Picked::Segment(picked) => match picked {
                    SegmentPicked::Point(picked) => {
                        ui.label(format!("Point number {}", picked));
                    }
                    SegmentPicked::Edge(picked) => {
                        ui.label(format!("Edge number {}", picked));
                    }
                },
            }
            let enabled = match picked {
                Picked::Surface(SurfacePicked::Vertex(_))
                | Picked::PointCloud(_)
                | Picked::Segment(SegmentPicked::Point(_)) => true,
                _ => false, //self.show_gizmo = false,
            };
            ui.add_enabled(
                enabled,
                Checkbox::new(&mut self.show_gizmo, "Edition Gizmo"),
            );
        }
    }

    pub(crate) fn draw_gizmo(
        &mut self,
        ui: &mut egui::Ui,
        view: glam::Mat4,
        proj: glam::Mat4,
        queue: &wgpu::Queue,
        surfaces: &mut IndexMap<String, crate::surface::geometry::DisplaySurface>,
        clouds: &mut IndexMap<String, crate::point_cloud::DisplayPointCloud>,
        curves: &mut IndexMap<String, crate::segment::DisplaySegment>,
        gizmo_hovered: &mut bool,
    ) -> bool {
        if self.show_gizmo {
            let viewport = ui.clip_rect();
            let view_m = view.as_dmat4();
            let proj_m = proj.as_dmat4();
            self.gizmo.update_config(GizmoConfig {
                view_matrix: view_m.into(),
                projection_matrix: proj_m.into(),
                modes: GizmoMode::all_translate().difference(enum_set!(GizmoMode::TranslateView)),
                orientation: GizmoOrientation::Local,
                viewport,
                ..Default::default()
            });

            let interacted = match self.picked_item.as_ref() {
                Some((name, Picked::Surface(SurfacePicked::Vertex(v)))) => surfaces
                    .get_mut(name)
                    .map(|surface| {
                        let pos = surface.geometry().get_vertex_pos(*v);
                        let transform = surface.transform.get_local_transform(pos);
                        if let Some((_result, new_transforms)) =
                            self.gizmo.interact(ui, &[transform])
                        {
                            let new_pos = surface
                                .transform
                                .reverse_local_transform(new_transforms[0].translation.into());
                            surface.move_vertex(queue, *v, new_pos);
                            true
                        } else {
                            false
                        }
                    })
                    .or(Some(false))
                    .unwrap(),
                Some((name, Picked::PointCloud(v))) => clouds
                    .get_mut(name)
                    .map(|cloud| {
                        let pos = cloud.geometry().get_vertex_pos(*v);
                        let transform = cloud.transform.get_local_transform(pos);
                        if let Some((_result, new_transforms)) =
                            self.gizmo.interact(ui, &[transform])
                        {
                            let new_pos = cloud
                                .transform
                                .reverse_local_transform(new_transforms[0].translation.into());
                            cloud.move_vertex(queue, *v, new_pos);
                            true
                        } else {
                            false
                        }
                    })
                    .or(Some(false))
                    .unwrap(),
                Some((name, Picked::Segment(SegmentPicked::Point(v)))) => curves
                    .get_mut(name)
                    .map(|curve| {
                        let pos = curve.geometry().get_vertex_pos(*v);
                        let transform = curve.transform.get_local_transform(pos);
                        if let Some((_result, new_transforms)) =
                            self.gizmo.interact(ui, &[transform])
                        {
                            let new_pos = curve
                                .transform
                                .reverse_local_transform(new_transforms[0].translation.into());
                            curve.move_vertex(queue, *v, new_pos);
                            true
                        } else {
                            false
                        }
                    })
                    .or(Some(false))
                    .unwrap(),
                _ => false,
            };
            *gizmo_hovered |= self.gizmo.is_focused();
            interacted
        } else {
            false
        }
    }
}
