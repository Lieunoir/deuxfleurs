use crate::ui::UiDataElement;
use egui::Shape::Path;
use egui::{Color32, Pos2, Stroke};
use egui_plot::{Bar, BarChart, CoordinatesFormatter, Plot};
use epaint::PathShape;

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
    Twilight,
    BrBG,
    RdBU,
}

#[derive(Clone, PartialEq)]
pub struct ColorMap {
    pub colors: Colors,
    bars: Vec<Bar>,
    min: f64,
    max: f64,
}

fn get_bars(values: &[f32], n_bars: usize) -> Vec<Bar> {
    let (min, max) = values
        .iter()
        .fold((f32::MAX, f32::MIN), |(min, max), value| {
            (min.min(*value), max.max(*value))
        });
    let mut histogram = vec![0; n_bars];
    for v in values {
        let clamped = (v - min) / (max - min);
        let clamped = (clamped * (n_bars as f32)) as usize;
        histogram[clamped.min(n_bars - 1)] += 1;
    }
    histogram
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            Bar::new((min + (max - min) * i as f32) as f64, s as f64).width((max - min) as f64)
        })
        .collect()
}

impl ColorMap {
    pub(crate) fn new(values: &[f32]) -> Self {
        let mut res = Self {
            bars: get_bars(values, 120),
            colors: Colors::Viridis,
            min: 0.,
            max: 1.,
        };
        res.apply_bar_colors();
        res
    }

    fn apply_bar_colors(&mut self) {
        let color_map = self.colors;
        let n_values = self.bars.len();
        for (i, bar) in self.bars.iter_mut().enumerate() {
            let value =
                (-self.min as f32 + i as f32 / n_values as f32) / (self.max - self.min) as f32;
            let opacify = if value < 0. || value > 1. { 0.6 } else { 1. };
            let value = value.max(0.);
            let value = value.min(1.);
            let colors = color_map.compute_color(value);
            let bar_c = bar.clone();
            let color = Color32::from_rgb(
                (colors[0] * opacify * 255.) as u8,
                (colors[1] * opacify * 255.) as u8,
                (colors[2] * opacify * 255.) as u8,
            );
            *bar = bar_c.fill(color);
        }
    }

    pub(crate) fn get_value(&self) -> ColorMapValues {
        ColorMapValues {
            colors: self.colors.get_value(),
            min: self.min as f32,
            max: self.max as f32,
            _pad: [0; 2],
        }
    }

    pub(crate) fn recycle(&mut self, other: Self) {
        self.min = other.min;
        self.max = other.max;
        self.colors = other.colors;
        self.apply_bar_colors();
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
            0.25664562, 1.1613264, -14.784454, 76.28986, -217.01733, 326.72885, -238.12221,
            66.48359,
        ],
        green: [
            0.0039559007,
            1.5036448,
            -1.2557336,
            2.469486,
            -8.550757,
            18.999912,
            -19.025986,
            6.762205,
        ],
        blue: [
            0.32293624, 1.9620758, -7.801855, 24.8746, -65.22319, 110.47881, -100.77647, 36.299873,
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

    pub const TWILIGHT: ColorsValues = ColorsValues {
        red: [
            0.990215, -4.365588, 3.055973, 81.3124, -403.97992, 803.4978, -716.5665, 237.09198,
        ],
        green: [
            0.9360863, -3.0730484, 23.01469, -143.63342, 386.0892, -500.5965, 319.16608, -80.99168,
        ],
        blue: [
            1.0401875, -7.757017, 82.682014, -390.78943, 884.20605, -1036.1959, 610.2725,
            -142.44864,
        ],
    };

    pub const BRBG: ColorsValues = ColorsValues {
        red: [
            0.328325,
            2.3692522,
            0.014967918,
            -19.132025,
            87.46912,
            -192.46042,
            181.94604,
            -60.535843,
        ],
        green: [
            0.18887857, 0.90392935, 0.97115517, 37.013916, -158.98076, 250.70457, -181.76689,
            51.20108,
        ],
        blue: [
            0.019865572,
            1.0891378,
            -23.580093,
            193.56819,
            -536.8783,
            683.78796,
            -416.8518,
            99.03285,
        ],
    };

    pub const RDBU: ColorsValues = ColorsValues {
        red: [
            0.40488827, 3.4446564, -2.9528008, -46.060135, 229.68684, -458.21353, 405.41907,
            -131.70804,
        ],
        green: [
            -0.0004091859,
            -0.84625244,
            24.876186,
            -80.24888,
            150.61618,
            -200.69029,
            155.6336,
            -49.15227,
        ],
        blue: [
            0.121104434,
            1.2048302,
            -15.056142,
            104.97905,
            -231.92496,
            193.28596,
            -34.771202,
            -17.459175,
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
            Colors::Twilight => Colors::TWILIGHT,
            Colors::BrBG => Colors::BRBG,
            Colors::RdBU => Colors::RDBU,
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
            Colors::Twilight => "Twilight",
            Colors::BrBG => "BrBG",
            Colors::RdBU => "RdBU",
        }
    }

    fn compute_color(&self, x: f32) -> [f32; 3] {
        //let x = (clamp(dist, data_uniform.min, data_uniform.max) - data_uniform.min) / (data_uniform.max - data_uniform.min);
        let ColorsValues { red, green, blue } = self.get_value();
        let polynomial = [
            1.0,
            x,
            x * x,
            x * x * x,
            x * x * x * x,
            x * x * x * x * x,
            x * x * x * x * x * x,
            x * x * x * x * x * x * x,
        ];
        //let v4: vec4<f32> = vec4<f32>(1.0, x, x*x, x*x*x);
        //let v2: vec4<f32> = v4 * v4.w * x;
        let red = polynomial
            .iter()
            .zip(red)
            .fold(0., |acc, (c, v)| acc + (c * v));
        let green = polynomial
            .iter()
            .zip(green)
            .fold(0., |acc, (c, v)| acc + (c * v));
        let blue = polynomial
            .iter()
            .zip(blue)
            .fold(0., |acc, (c, v)| acc + (c * v));
        [red, green, blue]
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
            radius,
            fill: visuals.bg_fill,
            stroke: visuals.fg_stroke,
        });
        ui.painter().add(epaint::CircleShape {
            center: ub_pos,
            radius,
            fill: visuals.bg_fill,
            stroke: visuals.fg_stroke,
        });
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
                    .selectable_value(&mut self.colors, Colors::Turbo, Colors::Turbo.get_name())
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.colors,
                        Colors::Viridis,
                        Colors::Viridis.get_name(),
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.colors,
                        Colors::Inferno,
                        Colors::Inferno.get_name(),
                    )
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::Magma, Colors::Magma.get_name())
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::Plasma, Colors::Plasma.get_name())
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::BrBG, Colors::BrBG.get_name())
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.colors, Colors::RdBU, Colors::RdBU.get_name())
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.colors,
                        Colors::Coolwarm,
                        Colors::Coolwarm.get_name(),
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.colors,
                        Colors::Twilight,
                        Colors::Twilight.get_name(),
                    )
                    .changed();
            });
        if changed {
            self.apply_bar_colors();
        }

        Plot::new("plot")
            .label_formatter(|_, _| "".to_owned())
            .width(200.)
            .show_y(false)
            .show_x(true)
            .allow_zoom(false)
            .allow_boxed_zoom(false)
            .allow_scroll(false)
            .allow_drag(false)
            .clamp_grid(true)
            .show_axes(false)
            .show_grid(false)
            .view_aspect(4.)
            .coordinates_formatter(
                egui_plot::Corner::LeftTop,
                CoordinatesFormatter::new(|c, _| format!("{:.4}", c.x)),
            )
            .show(ui, |ui| {
                ui.bar_chart(BarChart::new(self.bars.clone()).allow_hover(false));
            });
        ui.horizontal(|ui| {
            if windowing_ui(ui, &100., &100., 0., 1., &mut self.min, &mut self.max).changed() {
                changed = true;
                self.apply_bar_colors();
            }
            ui.label("Range");
        });
        changed
    }
}
