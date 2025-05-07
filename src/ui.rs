use crate::{RunningState, StateHandle};
use egui::style::{WidgetVisuals, Widgets};
use egui::{Color32, Stroke};
use egui::{Response, Shadow, Widget};
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit::State;
use epaint::CornerRadiusF32;
use indexmap::IndexMap;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

/// When clicked, pops an interative window to load a mesh
pub struct LoadObjButton<'a, 'b, 'c> {
    name: &'a str,
    mesh_name: &'b str,
    state: &'c mut RunningState,
}

impl<'a, 'b, 'c> LoadObjButton<'a, 'b, 'c> {
    pub fn new(name: &'a str, mesh_name: &'b str, state: &'c mut RunningState) -> Self {
        Self {
            name,
            mesh_name,
            state,
        }
    }
}

impl<'a, 'b, 'c> Widget for LoadObjButton<'a, 'b, 'c> {
    fn ui(self, ui: &mut egui::Ui) -> Response {
        let button = egui::Button::new(self.name);
        let response = button.ui(ui);
        if response.clicked() {
            self.state.send_mesh(self.mesh_name.into());
        }
        response
    }
}

pub(crate) trait UiDataElement {
    fn draw(&mut self, ui: &mut egui::Ui, property_changed: &mut bool) -> bool;

    fn draw_gizmo(
        &mut self,
        _ui: &mut egui::Ui,
        _view: cgmath::Matrix4<f32>,
        _proj: cgmath::Matrix4<f32>,
        _gizmo_hovered: &mut bool,
    ) -> bool {
        false
    }
}

pub(crate) struct UI {
    rpass: Renderer,
    ctx: egui::Context,
    state: State,
    // used to pass the output at each frame without overloading the render loop
    // could be better
    platform_output: Option<egui::output::PlatformOutput>,
    pub(crate) hovered: bool,
}

fn blue_visuals() -> egui::Visuals {
    egui::Visuals {
        window_fill: egui::Color32::from_rgba_premultiplied(12, 0, 70, 220),
        window_stroke: egui::Stroke::NONE,
        extreme_bg_color: egui::Color32::from_rgba_premultiplied(10, 0, 50, 240),
        faint_bg_color: egui::Color32::from_rgba_premultiplied(10, 0, 50, 200),
        window_corner_radius: CornerRadiusF32::same(1.0).into(),
        window_highlight_topmost: false,
        window_shadow: Shadow::NONE,
        widgets: Widgets {
            noninteractive: WidgetVisuals {
                weak_bg_fill: Color32::from_gray(27),
                bg_fill: Color32::from_gray(27),
                bg_stroke: Stroke::new(1.0, egui::Color32::from_black_alpha(200)), // separators, indentation lines
                fg_stroke: Stroke::new(1.0, Color32::from_gray(240)), // normal text color
                corner_radius: CornerRadiusF32::same(2.0).into(),
                expansion: 0.0,
            },
            inactive: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(160), // button background
                bg_fill: egui::Color32::from_black_alpha(160),      // checkbox background
                bg_stroke: Default::default(),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(240)), // button text
                corner_radius: CornerRadiusF32::same(2.0).into(),
                expansion: 0.0,
            },
            hovered: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(180), // button background
                bg_fill: egui::Color32::from_black_alpha(180),      // checkbox background
                bg_stroke: Stroke::new(1.0, Color32::from_gray(250)), // e.g. hover over window edge or button
                fg_stroke: Stroke::new(1.5, Color32::from_gray(250)),
                corner_radius: CornerRadiusF32::same(3.0).into(),
                expansion: 1.0,
            },
            active: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(150),
                bg_fill: egui::Color32::from_black_alpha(150),
                bg_stroke: Stroke::new(1.0, Color32::WHITE),
                fg_stroke: Stroke::new(2.0, Color32::WHITE),
                corner_radius: CornerRadiusF32::same(2.0).into(),
                expansion: 1.0,
            },
            open: WidgetVisuals {
                weak_bg_fill: Color32::from_gray(45),
                bg_fill: Color32::from_gray(27),
                bg_stroke: Stroke::new(1.0, Color32::from_gray(210)),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(210)),
                corner_radius: CornerRadiusF32::same(2.0).into(),
                expansion: 0.0,
            },
        },
        ..Default::default()
    }
}

fn transparent_visuals() -> egui::Visuals {
    egui::Visuals {
        window_fill: egui::Color32::from_black_alpha(220),
        window_stroke: egui::Stroke::NONE,
        extreme_bg_color: egui::Color32::from_black_alpha(220),
        faint_bg_color: egui::Color32::from_black_alpha(100),
        window_corner_radius: CornerRadiusF32::same(1.0).into(),

        window_highlight_topmost: false,
        window_shadow: Shadow::NONE,
        widgets: Widgets {
            noninteractive: WidgetVisuals {
                weak_bg_fill: Color32::from_gray(27),
                bg_fill: Color32::from_gray(27),
                bg_stroke: Stroke::new(1.0, egui::Color32::from_black_alpha(80)), // separators, indentation lines
                fg_stroke: Stroke::new(1.0, Color32::from_gray(180)), // normal text color
                corner_radius: CornerRadiusF32::same(2.0).into(),
                expansion: 0.0,
            },
            inactive: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(100), // button background
                bg_fill: egui::Color32::from_black_alpha(100),      // checkbox background
                bg_stroke: Default::default(),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(200)), // button text
                corner_radius: CornerRadiusF32::same(2.0).into(),
                expansion: 0.0,
            },
            hovered: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(110), // button background
                bg_fill: egui::Color32::from_black_alpha(110),      // checkbox background
                bg_stroke: Stroke::new(1.0, Color32::from_gray(200)), // e.g. hover over window edge or button
                fg_stroke: Stroke::new(1.5, Color32::from_gray(220)),
                corner_radius: CornerRadiusF32::same(3.0).into(),
                expansion: 1.0,
            },
            active: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(95),
                bg_fill: egui::Color32::from_black_alpha(95),
                bg_stroke: Stroke::new(1.0, Color32::WHITE),
                fg_stroke: Stroke::new(2.0, Color32::WHITE),
                corner_radius: CornerRadiusF32::same(2.0).into(),
                expansion: 1.0,
            },
            open: WidgetVisuals {
                weak_bg_fill: Color32::from_gray(45),
                bg_fill: Color32::from_gray(27),
                bg_stroke: Stroke::new(1.0, Color32::from_gray(60)),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(210)),
                corner_radius: CornerRadiusF32::same(2.0).into(),
                expansion: 0.0,
            },
        },
        ..Default::default()
    }
}

impl UI {
    pub(crate) fn new(
        device: &wgpu::Device,
        event_loop: &ActiveEventLoop,
        target_format: wgpu::TextureFormat,
        scale_factor: f64,
    ) -> Self {
        let rpass = Renderer::new(device, target_format, None, 1, true);
        let ctx = egui::Context::default();
        //TODO some kind of styling
        let visuals = blue_visuals();
        ctx.set_style(egui::Style {
            animation_time: 0.,
            ..Default::default()
        });
        ctx.set_visuals(visuals);
        /*
        let mut style: egui::Style = (*ctx.style()).clone();
        style.override_font_id = Some(egui::FontId::proportional(20.));
        ctx.set_style(style);
        */
        let state = State::new(
            ctx.clone(),
            egui::viewport::ViewportId::ROOT,
            event_loop,
            Some(scale_factor as f32),
            None,
            None,
        );
        Self {
            rpass,
            ctx,
            state,
            platform_output: None,
            hovered: false,
        }
    }

    pub(crate) fn process_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    pub(crate) fn draw_models(
        &mut self,
        window: &Window,
        surfaces: &mut IndexMap<String, crate::surface::DisplaySurface>,
        clouds: &mut IndexMap<String, crate::point_cloud::DisplayPointCloud>,
        curves: &mut IndexMap<String, crate::segment::DisplaySegment>,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
    ) {
        let input = self.state.take_egui_input(window);
        self.ctx.begin_pass(input);
        self.hovered = false;

        if let Some(response) = egui::Window::new("Models")
            .anchor(egui::Align2::LEFT_TOP, [5., 5.])
            .resizable(false)
            .vscroll(true)
            .default_width(270.)
            .min_height(650.)
            .show(&self.ctx, |ui| {
                for (name, surface) in surfaces.iter_mut() {
                    let label = format!(
                        "{}: {} vertices, {} faces",
                        name,
                        surface.element.geometry().vertices.len(),
                        surface.element.geometry().indices.size(),
                    );
                    let id = ui.make_persistent_id(label.clone());
                    egui::collapsing_header::CollapsingState::load_with_default_open(
                        ui.ctx(),
                        id,
                        true,
                    )
                    .show_header(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut surface.element.show, label).changed() {
                                surface.element.updater.dirty = true;
                            }
                        })
                    })
                    .body(|ui| {
                        surface.draw_ui(ui);
                    });
                }

                for (name, cloud) in clouds.iter_mut() {
                    let label = format!(
                        "{} : {} points",
                        name,
                        cloud.element.geometry().positions.len(),
                    );
                    let id = ui.make_persistent_id(label.clone());
                    egui::collapsing_header::CollapsingState::load_with_default_open(
                        ui.ctx(),
                        id,
                        true,
                    )
                    .show_header(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut cloud.element.show, label).changed() {
                                cloud.element.updater.dirty = true;
                            }
                        })
                    })
                    .body(|ui| {
                        cloud.draw_ui(ui);
                    });
                }

                for (name, curve) in curves.iter_mut() {
                    let label = format!(
                        "{} : {} points, {} edges",
                        name,
                        curve.element.geometry().positions.len(),
                        curve.element.geometry().connections.len(),
                    );
                    let id = ui.make_persistent_id(label.clone());
                    egui::collapsing_header::CollapsingState::load_with_default_open(
                        ui.ctx(),
                        id,
                        true,
                    )
                    .show_header(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut curve.element.show, label).changed() {
                                curve.element.updater.dirty = true;
                            }
                        })
                    })
                    .body(|ui| {
                        curve.draw_ui(ui);
                    });
                }
            })
        {
            self.hovered |= response.response.contains_pointer();
        }
        egui::Area::new("Viewport".into())
            .fixed_pos((0.0, 0.0))
            .show(&self.ctx, |ui| {
                for (_, surface) in surfaces.iter_mut() {
                    surface.draw_gizmo(ui, view, proj, &mut self.hovered);
                }
                for (_, curve) in curves.iter_mut() {
                    curve.draw_gizmo(ui, view, proj, &mut self.hovered);
                }
                for (_, cloud) in clouds.iter_mut() {
                    cloud.draw_gizmo(ui, view, proj, &mut self.hovered);
                }
            });
    }

    pub(crate) fn draw_callback<T: FnMut(&mut egui::Ui, &mut RunningState)>(
        &mut self,
        state: &mut RunningState,
        callback: &mut T,
    ) {
        if let Some(response) = egui::Window::new("Interactions")
            .anchor(egui::Align2::RIGHT_TOP, [-5., 5.])
            .resizable(false)
            .show(&self.ctx, |ui| {
                if ui.add(egui::Button::new("Fit camera")).clicked() {
                    state.resize_scene();
                }
                if ui.add(egui::Button::new("Screenshot")).clicked() {
                    state.screenshot();
                }
                if let Some((picked_name, picked_number)) = state.get_picked() {
                    let picked_number = *picked_number;
                    ui.label(format!("Picked {}", picked_name));
                    if let Some(surface) = state.get_surface(&picked_name) {
                        surface.draw_element_info(picked_number, ui);
                    } else if let Some(cloud) = state.get_point_cloud(&picked_name) {
                        cloud.draw_element_info(picked_number, ui);
                    } else if let Some(curve) = state.get_segment(&picked_name) {
                        curve.draw_element_info(picked_number, ui);
                    }
                }

                callback(ui, state)
            })
        {
            self.hovered |= response.response.contains_pointer()
        }
    }

    // not sure about returning a tuple here
    pub(crate) fn render_deltas(
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
        let full_output = self.ctx.end_pass();
        let textures_delta = full_output.textures_delta;
        let clipped_primitives = self
            .ctx
            .tessellate(full_output.shapes, self.ctx.pixels_per_point());

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: self.ctx.pixels_per_point(),
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

    pub(crate) fn render(
        &self,
        mut render_pass: wgpu::RenderPass<'static>,
        clipped_primitives: &[egui::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
    ) {
        self.rpass
            .render(&mut render_pass, clipped_primitives, screen_descriptor);
    }

    pub(crate) fn handle_platform_output(&mut self, window: &Window) {
        if let Some(platform_output) = self.platform_output.take() {
            self.state.handle_platform_output(window, platform_output);
        }
    }
}
