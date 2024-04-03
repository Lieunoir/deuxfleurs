use crate::ui::UiDataElement;
use egui::Shape::Path;
use egui::{Pos2, Stroke};
use epaint::PathShape;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorsValues {
    pub red: [f32; 8],
    pub green: [f32; 8],
    pub blue: [f32; 8],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorMapValues {
    colors: ColorsValues,
    min: f32,
    max: f32,
    _pad: [u32; 2],
}

#[derive(Copy, Clone, PartialEq)]
pub enum Colors {
    Turbo,
    Viridis,
    Inferno,
    Magma,
    Plasma,
    Coolwarm,
}

#[derive(Copy, Clone, PartialEq)]
pub struct ColorMap {
    colors: Colors,
    min: f64,
    max: f64,
}

impl Default for ColorMap {
    fn default() -> Self {
        Self {
            colors: Colors::Plasma,
            min: 0.,
            max: 1.,
        }
    }
}

impl ColorMap {
    pub fn get_value(&self) -> ColorMapValues {
        ColorMapValues {
            colors: self.colors.get_value(),
            min: self.min as f32,
            max: self.max as f32,
            _pad: [0; 2],
        }
    }
}

impl Colors {
    // from https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
    pub const TURBO: ColorsValues = ColorsValues {
        red: [
            0.13572138,
            4.61539260,
            -42.66032258,
            132.13108234,
            -152.94239396,
            59.28637943,
            0.,
            0.,
        ],
        green: [
            0.09140261,
            2.19418839,
            4.84296658,
            -14.18503333,
            4.27729857,
            2.829566004,
            0.,
            0.,
        ],
        blue: [
            0.10667330,
            12.64194608,
            -60.58204836,
            110.36276771,
            -89.90310912,
            27.34824973,
            0.,
            0.,
        ],
    };
    pub const VIRIDIS: ColorsValues = ColorsValues {
        red: [
            0.26626816, 3.7648385, -60.260567, 359.6886, -1049.4519, 1574.2263, -1160.1252,
            332.88736,
        ],
        green: [
            0.0037321262,
            2.7245293,
            -26.431007,
            166.67218,
            -510.38892,
            799.1755,
            -614.4233,
            183.57681,
        ],
        blue: [
            0.32955983, -0.4777524, 29.683285, -197.1112, 575.40265, -845.7034, 606.9172,
            -168.89499,
        ],
    };
    pub const INFERNO: ColorsValues = ColorsValues {
        red: [
            -0.00016338378,
            -0.073518544,
            14.817125,
            -61.378624,
            133.81487,
            -154.57034,
            85.55296,
            -17.174284,
        ],
        green: [
            0.00032205507,
            1.491896,
            -18.942537,
            105.34187,
            -280.27472,
            391.42755,
            -272.43903,
            74.39492,
        ],
        blue: [
            0.015863627,
            0.21556938,
            38.829773,
            -265.4191,
            773.58936,
            -1158.4982,
            864.58246,
            -252.67209,
        ],
    };

    pub const MAGMA: ColorsValues = ColorsValues {
        red: [
            0.00021596998,
            -0.5137123,
            20.579655,
            -99.91494,
            256.9034,
            -350.63345,
            237.50842,
            -62.941208,
        ],
        green: [
            -8.404814e-5,
            2.0975368,
            -27.284979,
            155.73149,
            -429.42642,
            616.1145,
            -438.69254,
            122.452614,
        ],
        blue: [
            0.014895931,
            0.71660554,
            25.154245,
            -151.14711,
            386.30112,
            -525.287,
            370.7574,
            -105.76203,
        ],
    };

    pub const PLASMA: ColorsValues = ColorsValues {
        red: [
            0.05102612, 2.7417765, -10.374987, 47.75827, -122.86391, 167.50691, -115.23589,
            31.357975,
        ],
        green: [
            0.031248435,
            0.56726646,
            -14.215471,
            85.1849,
            -208.33847,
            259.38776,
            -162.15656,
            40.515724,
        ],
        blue: [
            0.5295401, 0.6883895, 4.760893, -39.355995, 92.17261, -102.47303, 54.79296, -10.985884,
        ],
    };

    pub const COOLWARM: ColorsValues = ColorsValues {
        red: [
            0.23141083,
            1.0209837,
            2.7073274,
            -12.625891,
            36.274986,
            -58.640087,
            44.20847,
            -12.4713125,
        ],
        green: [
            0.2983393,
            1.7745266,
            -0.079018354,
            -8.051756,
            36.45388,
            -87.40499,
            90.02061,
            -32.984547,
        ],
        blue: [
            0.7521305, 1.7568098, -4.6975117, 14.51453, -48.73433, 75.045944, -52.16454, 13.676519,
        ],
    };

    pub fn get_value(&self) -> ColorsValues {
        match self {
            Colors::Turbo => Colors::TURBO,
            Colors::Viridis => Colors::VIRIDIS,
            Colors::Inferno => Colors::INFERNO,
            Colors::Magma => Colors::MAGMA,
            Colors::Plasma => Colors::PLASMA,
            Colors::Coolwarm => Colors::COOLWARM,
        }
    }

    pub fn get_name(&self) -> &str {
        match self {
            Colors::Turbo => "Turbo",
            Colors::Viridis => "Viridis",
            Colors::Inferno => "Inferno",
            Colors::Magma => "Magma",
            Colors::Plasma => "Plasma",
            Colors::Coolwarm => "Coolwarm",
        }
    }
}

pub fn windowing_ui(
    ui: &mut egui::Ui,
    width: &f64,
    height: &f64,
    min: f64,
    max: f64,
    lb: &mut f64,
    ub: &mut f64,
) -> egui::Response {
    let desired_size = egui::vec2(*width as f32, *height as f32 / 10.0);
    let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::drag());
    let range = max - min;
    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&response);
        let bar_line = vec![rect.left_center(), rect.right_center()];
        ui.painter().add(Path(PathShape::line(
            bar_line,
            Stroke::new(*height as f32 / 20.0, visuals.bg_fill),
        )));
        let mut bounds_i = [0.0; 2];
        bounds_i[0] = *lb / range * width;
        bounds_i[1] = *ub / range * width;
        let mut lb_pos = Pos2 {
            x: rect.left_center().x + bounds_i[0] as f32,
            y: rect.left_center().y,
        };
        let mut ub_pos = Pos2 {
            x: rect.left_center().x + bounds_i[1] as f32,
            y: rect.right_center().y,
        };

        if response.dragged() {
            let pos = response.interact_pointer_pos();

            match pos {
                None => {}
                Some(p) => {
                    let dist_1 = (lb_pos.x - p.x).abs();
                    let dist_2 = (ub_pos.x - p.x).abs();
                    if dist_1 < dist_2 {
                        // dragging the lower one
                        lb_pos.x = f32::min(f32::max(p.x, rect.left_center().x), ub_pos.x);
                        response.mark_changed();
                    } else {
                        // dragging the upper one
                        ub_pos.x = f32::max(f32::min(p.x, rect.right_center().x), lb_pos.x);
                        response.mark_changed();
                    }
                }
            }
        }

        bounds_i[0] = lb_pos.x as f64 - rect.left_center().x as f64;
        *lb = bounds_i[0] / width * range;
        bounds_i[1] = ub_pos.x as f64 - rect.left_center().x as f64;
        *ub = bounds_i[1] / width * range;

        let bar_line = vec![lb_pos, ub_pos];
        ui.painter().add(Path(PathShape::line(
            bar_line,
            Stroke::new(*height as f32 / 20.0, visuals.fg_stroke.color),
        )));
        let radius = *width as f32 / 20.;
        ui.painter().add(epaint::CircleShape {
            center: lb_pos,
            radius: radius,
            fill: visuals.bg_fill,
            stroke: visuals.fg_stroke,
        });
        ui.painter().add(epaint::CircleShape {
            center: ub_pos,
            radius: radius,
            fill: visuals.bg_fill,
            stroke: visuals.fg_stroke,
        });

        /*
        ui.vertical_centered(|ui| {
            ui.horizontal(|ui| {
                ui.add(
                    DragValue::new( lb)
                        .clamp_range(min..=*ub),
                );
                //ui.add_space(2.0 * ui.available_width() - *width as f32 / 0.85);
                ui.add(
                    DragValue::new( ub)
                        .clamp_range(*lb..=max),
                );
            });
        });*/
    }
    response
}

impl UiDataElement for ColorMap {
    fn draw(&mut self, ui: &mut egui::Ui, _property_changed: &mut bool) -> bool {
        let mut changed = false;
        egui::ComboBox::from_label("ColorMap")
            .selected_text(self.colors.get_name())
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::Turbo, "Turbo")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::Viridis, "Viridis")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::Inferno, "Inferno")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::Magma, "Magma")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::Plasma, "Plasma")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::Coolwarm, "Coolwarm")
                    .changed();
            });
        ui.horizontal(|ui| {
            changed |=
                windowing_ui(ui, &100., &100., 0., 1., &mut self.min, &mut self.max).changed();
            ui.label("Range");
        });
        changed
    }
}

impl ColorMapValues {
    pub fn build_buffer(self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Data buffer"),
            contents: bytemuck::cast_slice(&[self]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
}
