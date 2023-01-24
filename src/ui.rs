use egui_gizmo::GizmoMode;
use egui_wgpu::{renderer::ScreenDescriptor, Renderer};
use egui_winit::State;
use std::collections::HashMap;
use winit::event_loop::EventLoop;
use winit::window::Window;

use crate::model::data::MeshData;

pub trait UiMeshDataElement {
    fn draw(&mut self, ui: &mut egui::Ui) -> bool;
}

pub struct UI {
    rpass: Renderer,
    ctx: egui::Context,
    state: State,
    // used to pass the output at each frame without overloading the render loop
    // could be better
    platform_output: Option<egui::output::PlatformOutput>,
}

impl UI {
    pub fn new<T>(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        event_loop: &EventLoop<T>,
    ) -> Self {
        let rpass = Renderer::new(
            device,
            target_format,
            Some(crate::texture::Texture::DEPTH_FORMAT),
            1,
        );
        let ctx = egui::Context::default();
        //TODO some kind of styling
        let visuals = egui::Visuals {
            window_fill: egui::Color32::from_rgba_premultiplied(10, 0, 50, 192),
            ..Default::default()
        };
        ctx.set_visuals(visuals);
        /*
        let mut style: egui::Style = (*ctx.style()).clone();
        style.override_font_id = Some(egui::FontId::proportional(20.));
        ctx.set_style(style);
        */
        let state = State::new(event_loop);
        Self {
            rpass,
            ctx,
            state,
            platform_output: None,
        }
    }

    pub fn process_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.state.on_event(&self.ctx, event).consumed
    }

    pub fn draw_models(
        &mut self,
        window: &Window,
        models: &mut HashMap<String, crate::model::Model>,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
    ) {
        let input = self.state.take_egui_input(window);
        self.ctx.begin_frame(input);

        egui::Window::new("Models")
            .anchor(egui::Align2::LEFT_TOP, [5., 5.])
            .resizable(false)
            .show(&self.ctx, |ui| {
                for (name, model) in models.iter_mut() {
                    egui::CollapsingHeader::new(format!(
                        "{} : {} vertices, {} faces",
                        name,
                        model.mesh.vertices.len(),
                        model.mesh.indices.len()
                    ))
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut model.mesh.show, "Show");
                            let mut show_edges = model.mesh.show_edges;
                            ui.checkbox(&mut show_edges, "Edges");
                            model.mesh.show_edges(show_edges);
                            let mut smooth = model.mesh.smooth;
                            ui.checkbox(&mut smooth, "Smooth");
                            model.mesh.set_smooth(smooth);
                        });
                        let mut mesh_color = egui::Rgba::from_rgba_unmultiplied(
                            model.mesh.color[0],
                            model.mesh.color[1],
                            model.mesh.color[2],
                            model.mesh.color[3],
                        );
                        let mut picker_changed = false;
                        ui.horizontal(|ui| {
                            picker_changed = egui::widgets::color_picker::color_edit_button_rgba(
                                ui,
                                &mut mesh_color,
                                egui::widgets::color_picker::Alpha::Opaque,
                            )
                            .changed();
                            ui.label("Mesh color");
                        });
                        model.mesh.color = mesh_color.to_array();
                        if model.mesh.shown_data.is_none() && picker_changed {
                            model.mesh.uniform_changed = true;
                        }
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut model.mesh.show_gizmo, "Show Gizmo");
                            egui::ComboBox::from_label("Guizmo Mode")
                                .selected_text(format!("{:?}", model.mesh.gizmo_mode))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut model.mesh.gizmo_mode,
                                        GizmoMode::Rotate,
                                        "Rotate",
                                    );
                                    ui.selectable_value(
                                        &mut model.mesh.gizmo_mode,
                                        GizmoMode::Translate,
                                        "Translate",
                                    );
                                    ui.selectable_value(
                                        &mut model.mesh.gizmo_mode,
                                        GizmoMode::Scale,
                                        "Scale",
                                    );
                                });
                        });
                        for (name, field) in &mut model.mesh.vector_fields {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut field.shown, name.clone());
                            });
                        }
                        for (name, data) in &mut model.mesh.datas {
                            let active = model.mesh.shown_data == Some(name.clone());
                            ui.horizontal(|ui| {
                                let mut change_active = active;
                                ui.checkbox(&mut change_active, name.clone());
                                if change_active != active {
                                    if !active {
                                        model.mesh.data_to_show = Some(Some(name.clone()))
                                    } else {
                                        model.mesh.data_to_show = Some(None)
                                    }
                                }
                                match data {
                                    MeshData::UVMap(..) => {
                                        ui.label("UV Map");
                                    }
                                    MeshData::UVCornerMap(..) => {
                                        ui.label("UV Corner Map");
                                    }
                                    MeshData::VertexScalar(..) => {
                                        ui.label("Vertex Scalar");
                                    }
                                    MeshData::FaceScalar(..) => {
                                        ui.label("Face Scalar");
                                    }
                                    MeshData::Color(..) => {
                                        ui.label("Color");
                                    }
                                    _ => (),
                                };
                            });
                            egui::CollapsingHeader::new(name)
                                .default_open(false)
                                .show(ui, |ui| {
                                    let changed = data.draw(ui);
                                    if active && changed {
                                        model.mesh.uniform_changed = true;
                                    }
                                });
                        }
                    });
                }
                ui.allocate_space(ui.available_size());
            });
        egui::Area::new("Viewport")
            .fixed_pos((0.0, 0.0))
            .show(&self.ctx, |ui| {
                //Fix for not having a correct area
                //ui.with_layer_id(egui::LayerId::background(), |ui| {
                for (name, model) in models.iter_mut() {
                    if model.mesh.show_gizmo {
                        let gizmo = egui_gizmo::Gizmo::new(format!("{} gizmo", name))
                            .view_matrix(view)
                            .projection_matrix(proj)
                            .model_matrix(model.mesh.transform.0)
                            .viewport(ui.clip_rect())
                            .mode(model.mesh.gizmo_mode);
                        let last_gizmo_response = gizmo.interact(ui);

                        if let Some(gizmo_response) = last_gizmo_response {
                            model.mesh.transform.0 = gizmo_response.transform;
                            model.mesh.refresh_transform();
                        }
                    }
                }
                //});
            });
    }

    pub fn draw_callback(
        &mut self,
        event_loop_proxy: &winit::event_loop::EventLoopProxy<crate::UserEvent>,
        state: &mut crate::State,
        callback: &mut Option<Box<dyn FnMut(&mut egui::Ui, &mut crate::State)>>,
    ) {
        egui::Window::new("Interactions")
            .anchor(egui::Align2::RIGHT_TOP, [-5., 5.])
            .resizable(false)
            .show(&self.ctx, |ui| {
                if ui.add(egui::Button::new("Screenshot")).clicked() {
                    state.screenshot();
                }
                if ui.add(egui::Button::new("Load obj")).clicked() {
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("obj", &["obj"])
                        //.set_directory("/")
                        .pick_file();
                    let event_loop_proxy = event_loop_proxy.clone();
                    let f = async move {
                        let file = file.await;
                        if let Some(file_handle) = file {
                            let data = file_handle.read().await;
                            if let Ok((mesh_v, mesh_f)) =
                                crate::resources::load_preloaded_mesh(data).await
                            {
                                event_loop_proxy
                                    .send_event(crate::UserEvent::LoadMesh(mesh_v, mesh_f))
                                    .ok();
                            }
                        }
                    };
                    //TODO avoid pollster blocking here
                    #[cfg(not(target_arch = "wasm32"))]
                    pollster::block_on(f);
                    #[cfg(target_arch = "wasm32")]
                    wasm_bindgen_futures::spawn_local(f);
                }
                if let Some((picked_name, picked_number)) = state.get_picked() {
                    ui.label(format!("Picked model : {}", picked_name));
                    ui.label(format!("Picked item : {}", picked_number));
                }

                if let Some(callback) = callback {
                    callback(ui, state)
                }
            });
    }

    // not sure about returning a tuple here
    pub fn render_deltas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
    ) -> (
        Vec<wgpu::CommandBuffer>,
        Vec<egui::ClippedPrimitive>,
        ScreenDescriptor,
    ) {
        let full_output = self.ctx.end_frame();
        let textures_delta = full_output.textures_delta;
        let clipped_primitives = self.ctx.tessellate(full_output.shapes);

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: 1.,
        };

        let user_cmd_bufs = {
            for (id, image_delta) in &textures_delta.set {
                self.rpass.update_texture(device, queue, *id, image_delta);
            }
            self.rpass.update_buffers(
                device,
                queue,
                encoder,
                &clipped_primitives,
                &screen_descriptor,
            )
        };
        self.platform_output = Some(full_output.platform_output);
        (user_cmd_bufs, clipped_primitives, screen_descriptor)
    }

    pub fn render<'a, 'b: 'a>(
        &'b self,
        render_pass: &mut wgpu::RenderPass<'a>,
        clipped_primitives: &[egui::ClippedPrimitive],
        screen_descriptor: &'a ScreenDescriptor,
    ) {
        self.rpass
            .render(render_pass, clipped_primitives, screen_descriptor);
    }

    pub fn handle_platform_output(&mut self, window: &Window) {
        if let Some(platform_output) = self.platform_output.take() {
            self.state
                .handle_platform_output(window, &self.ctx, platform_output);
        }
    }
}
