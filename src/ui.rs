use crate::picker::Picker;
use crate::window::RunningState;
use egui::epaint::CornerRadiusF32;
use egui::style::{WidgetVisuals, Widgets};
use egui::text::LayoutJob;
use egui::{
    Align, Color32, CornerRadius, FontSelection, RichText, Shadow, Stroke, TextFormat, TextWrapMode,
};
#[cfg(feature = "surface_button")]
use egui::{Response, Widget};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use egui_winit::State;
use indexmap::IndexMap;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

#[cfg(feature = "surface_button")]
/// When clicked, pops an interative window to load a mesh
pub struct LoadSurfaceButton<'a, 'b, 'c> {
    name: &'a str,
    mesh_name: &'b str,
    state: &'c mut RunningState,
}

#[cfg(feature = "surface_button")]
impl<'a, 'b, 'c> LoadSurfaceButton<'a, 'b, 'c> {
    pub fn new(name: &'a str, mesh_name: &'b str, state: &'c mut RunningState) -> Self {
        Self {
            name,
            mesh_name,
            state,
        }
    }
}

#[cfg(feature = "surface_button")]
impl<'a, 'b, 'c> Widget for LoadSurfaceButton<'a, 'b, 'c> {
    fn ui(self, ui: &mut egui::Ui) -> Response {
        let button = egui::Button::new(self.name);
        let response = button.ui(ui);
        if response.clicked() {
            self.state.send_mesh(self.mesh_name.into());
        }
        response
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

fn oklch_to_linear(l: f32, c: f32, h: f32, alpha: f32) -> egui::Rgba {
    let a = c * h.cos();
    let b = c * h.sin();

    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    egui::Rgba::from_rgba_premultiplied(
        (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s) - 1. + alpha,
        (-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s) - 1. + alpha,
        (-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s) - 1. + alpha,
        alpha,
    )
}

fn blue_visuals() -> egui::Visuals {
    let base_extreme_color =
        oklch_to_linear(0.35, 0.035, 302. / 360. * 2. * std::f32::consts::PI, 0.95);
    let base_color = oklch_to_linear(0.39, 0.035, 302. / 360. * 2. * std::f32::consts::PI, 0.95);
    let base_color_2 = oklch_to_linear(0.41, 0.035, 302. / 360. * 2. * std::f32::consts::PI, 0.95);
    let brighter_color = oklch_to_linear(0.57, 0.03, 302. / 360. * 2. * std::f32::consts::PI, 0.9);
    let brighter_color_2 = oklch_to_linear(0.6, 0.04, 302. / 360. * 2. * std::f32::consts::PI, 0.9);
    let even_brighter_color =
        oklch_to_linear(0.62, 0.03, 302. / 360. * 2. * std::f32::consts::PI, 0.9);
    let even_brighter_color_2 =
        oklch_to_linear(0.63, 0.04, 302. / 360. * 2. * std::f32::consts::PI, 0.9);
    let darker_color = oklch_to_linear(0.5, 0.03, 302. / 360. * 2. * std::f32::consts::PI, 0.9);
    let darker_color_2 = oklch_to_linear(0.53, 0.04, 302. / 360. * 2. * std::f32::consts::PI, 0.9);
    let window_stroke = oklch_to_linear(0.55, 0.04, 302. / 360. * 2. * std::f32::consts::PI, 0.9);

    let selection_color = oklch_to_linear(0.65, 0.035, 62. / 360. * 2. * std::f32::consts::PI, 0.9);
    let corner_radius: CornerRadius = CornerRadiusF32::same(2.).into();
    egui::Visuals {
        window_fill: base_color.into(),
        //window_stroke: egui::Stroke::NONE,
        window_stroke: egui::Stroke::new(1.0, Color32::from(window_stroke)),
        extreme_bg_color: base_extreme_color.into(),
        faint_bg_color: base_color_2.into(),
        window_corner_radius: CornerRadiusF32::same(2.0).into(),
        window_highlight_topmost: false,
        window_shadow: Shadow::NONE,
        popup_shadow: Shadow::NONE,
        handle_shape: egui::style::HandleShape::Rect { aspect_ratio: 0.4 },
        collapsing_header_frame: true,
        selection: egui::style::Selection {
            bg_fill: selection_color.into(),
            stroke: Stroke {
                width: 1.,
                color: Color32::WHITE,
            },
        },

        widgets: Widgets {
            noninteractive: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_rgba_unmultiplied(180, 180, 180, 160),
                bg_fill: egui::Color32::from_rgba_premultiplied(50, 30, 70, 195),
                bg_stroke: Stroke::new(1.0, Color32::from(darker_color_2)), // separators, indentation lines
                fg_stroke: Stroke::new(1.0, Color32::from_gray(255)),       // normal text color
                corner_radius,
                expansion: 0.0,
            },
            inactive: WidgetVisuals {
                weak_bg_fill: brighter_color_2.into(),
                bg_fill: brighter_color.into(),
                bg_stroke: Stroke::new(0.0, Color32::from(base_color)),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(220)), // button text
                corner_radius,
                expansion: 0.0,
            },
            hovered: WidgetVisuals {
                weak_bg_fill: even_brighter_color_2.into(),
                bg_fill: even_brighter_color.into(),
                bg_stroke: Stroke::new(1.0, Color32::from_gray(230)), // e.g. hover over window edge or button
                fg_stroke: Stroke::new(1.5, Color32::WHITE),
                corner_radius: CornerRadiusF32::same(corner_radius.average() + 1.).into(),
                expansion: 1.0,
            },
            active: WidgetVisuals {
                weak_bg_fill: darker_color_2.into(),
                bg_fill: darker_color.into(),
                bg_stroke: Stroke::new(1.0, Color32::WHITE),
                fg_stroke: Stroke::new(2.0, Color32::WHITE),
                corner_radius: CornerRadiusF32::same(corner_radius.average() + 1.).into(),
                expansion: 1.0,
            },
            open: WidgetVisuals {
                weak_bg_fill: base_color.into(),
                bg_fill: base_color.into(),
                bg_stroke: Stroke::new(1.0, Color32::from_gray(210)),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(210)),
                corner_radius,
                expansion: 0.0,
            },
        },
        ..Default::default()
    }
}

fn format_header(ui: &mut egui::Ui, name: &str, infos: String) -> LayoutJob {
    let mut job = LayoutJob::default();
    RichText::new(name).append_to(&mut job, &ui.style(), FontSelection::Default, Align::Center);
    job.append("", 10., TextFormat::default());
    RichText::new(infos)
        .size(10.)
        .color(ui.style().visuals.weak_text_color())
        .append_to(&mut job, &ui.style(), FontSelection::Default, Align::BOTTOM);
    job
}

impl UI {
    pub(crate) fn new(
        device: &wgpu::Device,
        event_loop: &ActiveEventLoop,
        target_format: wgpu::TextureFormat,
        scale_factor: f64,
    ) -> Self {
        let options = RendererOptions {
            msaa_samples: 1,
            dithering: true,
            depth_stencil_format: None,
            predictable_texture_filtering: false,
        };
        let rpass = Renderer::new(device, target_format, options);
        let ctx = egui::Context::default();
        //TODO some kind of styling
        let visuals = blue_visuals();
        ctx.set_global_style(egui::Style {
            animation_time: 0.,
            wrap_mode: Some(TextWrapMode::Wrap),
            interaction: egui::style::Interaction {
                tooltip_delay: 0.,
                ..Default::default()
            },
            ..Default::default()
        });
        ctx.set_visuals(visuals);
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
        surfaces: &mut IndexMap<String, crate::surface::geometry::DisplaySurface>,
        clouds: &mut IndexMap<String, crate::point_cloud::DisplayPointCloud>,
        curves: &mut IndexMap<String, crate::segment::DisplaySegment>,
        picker: &mut Picker,
        view: glam::Mat4,
        proj: glam::Mat4,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        refresh_screen: &mut bool,
    ) {
        let input = self.state.take_egui_input(window);
        self.ctx.begin_pass(input);
        self.hovered = false;

        let screen_height = self.ctx.screen_rect().height();
        let factor = self.ctx.pixels_per_point();

        let space_between_section = 9.;

        egui::Window::new("Shapes")
            .anchor(egui::Align2::LEFT_TOP, [5., 5.])
            .resizable(false)
            .default_width(270.)
            .show(&self.ctx, |ui| {
                egui::ScrollArea::new([true, true])
                    .min_scrolled_height(screen_height - 60.)
                    .show(ui, |ui| {
                        ui.set_min_height(650. / factor);
                        ui.set_min_width(270.);
                        for (name, surface) in surfaces.iter_mut() {
                            let header = format_header(
                                ui,
                                name,
                                format!(
                                    "{} vertices, {} faces",
                                    surface.geometry().vertices.len(),
                                    surface.geometry().indices.size(),
                                ),
                            );
                            let id = ui.make_persistent_id(header.text.clone());
                            egui::collapsing_header::CollapsingState::load_with_default_open(
                                ui.ctx(),
                                id,
                                true,
                            )
                            .show_header(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut surface.show, header).changed() {
                                        *refresh_screen = true;
                                    }
                                })
                            })
                            .body(|ui| {
                                surface.draw_ui(
                                    ui,
                                    device,
                                    queue,
                                    camera_bind_group_layout,
                                    refresh_screen,
                                );
                            });
                            ui.add_space(space_between_section);
                        }

                        for (name, cloud) in clouds.iter_mut() {
                            let header = format_header(
                                ui,
                                name,
                                format!("{} points", cloud.geometry().positions.len(),),
                            );
                            let id = ui.make_persistent_id(header.text.clone());
                            egui::collapsing_header::CollapsingState::load_with_default_open(
                                ui.ctx(),
                                id,
                                true,
                            )
                            .show_header(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut cloud.show, header).changed() {
                                        *refresh_screen = true;
                                    }
                                })
                            })
                            .body(|ui| {
                                cloud.draw_ui(
                                    ui,
                                    device,
                                    queue,
                                    camera_bind_group_layout,
                                    refresh_screen,
                                );
                            });
                            ui.add_space(space_between_section);
                        }

                        for (name, curve) in curves.iter_mut() {
                            let header = format_header(
                                ui,
                                name,
                                format!(
                                    "{} points, {} edges",
                                    curve.geometry().positions.len(),
                                    curve.geometry().connections.len()
                                ),
                            );
                            let id = ui.make_persistent_id(header.text.clone());
                            egui::collapsing_header::CollapsingState::load_with_default_open(
                                ui.ctx(),
                                id,
                                true,
                            )
                            .show_header(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut curve.show, header).changed() {
                                        *refresh_screen = true;
                                    }
                                })
                            })
                            .body(|ui| {
                                curve.draw_ui(
                                    ui,
                                    device,
                                    queue,
                                    camera_bind_group_layout,
                                    refresh_screen,
                                );
                            });
                            ui.add_space(space_between_section);
                        }
                    })
            })
            .map(|response| {
                self.hovered |= response.response.contains_pointer();
            });

        egui::Area::new("Viewport".into())
            .fixed_pos((0.0, 0.0))
            .show(&self.ctx, |ui| {
                for (_, surface) in surfaces.iter_mut() {
                    surface.draw_gizmo(ui, view, proj, queue, &mut self.hovered, refresh_screen);
                }
                for (_, curve) in curves.iter_mut() {
                    curve.draw_gizmo(ui, view, proj, queue, &mut self.hovered, refresh_screen);
                }
                for (_, cloud) in clouds.iter_mut() {
                    cloud.draw_gizmo(ui, view, proj, queue, &mut self.hovered, refresh_screen);
                }
                *refresh_screen |= picker.draw_gizmo(
                    ui,
                    view,
                    proj,
                    queue,
                    surfaces,
                    clouds,
                    curves,
                    &mut self.hovered,
                );
            });
    }

    pub(crate) fn draw_callback<T: FnMut(&mut egui::Ui, &mut RunningState)>(
        &mut self,
        state: &mut RunningState,
        callback: &mut T,
    ) {
        let screen_height = self.ctx.screen_rect().height();
        //let factor = self.ctx.pixels_per_point();
        egui::Window::new("Interact")
            .anchor(egui::Align2::RIGHT_TOP, [-5., 5.])
            .resizable(false)
            //.scroll([false, true])
            .default_width(230.)
            .show(&self.ctx, |ui| {
                egui::ScrollArea::new([true, true])
                    .min_scrolled_height(screen_height - 60.)
                    .show(ui, |ui| {
                        ui.set_min_width(230.);
                        ui.horizontal(|ui| {
                            if ui.button("Fit camera").clicked() {
                                state.resize_scene();
                            }
                            if ui.button("Screenshot").clicked() {
                                state.screenshot();
                            }
                        });

                        #[cfg(feature = "saves")]
                        ui.horizontal(|ui| {
                            if ui.button("Save state").clicked() {
                                state.save();
                            }
                            if ui.button("Load state").clicked() {
                                state.load();
                            }
                        });

                        state.0.settings.draw_ui(ui, &mut state.0.dirty);
                        state.picker.draw_ui(ui);
                        ui.separator();

                        callback(ui, state)
                    })
            })
            .map(|response| self.hovered |= response.response.contains_pointer());
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
