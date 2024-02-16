use crate::ui::UiDataElement;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorMapValues {
    pub red: [f32; 8],
    pub green: [f32; 8],
    pub blue: [f32; 8],
}

#[derive(Copy, Clone, PartialEq)]
pub enum ColorMap {
    Turbo,
    Viridis,
    Inferno,
    Magma,
    Plasma,
    Coolwarm,
}

impl Default for ColorMap {
    fn default() -> Self {
        ColorMap::Plasma
    }
}

impl ColorMap {
    pub const LAYOUT_ENTRY: wgpu::BindGroupLayoutEntry = wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    // from https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
    pub const TURBO: ColorMapValues = ColorMapValues {
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
    pub const VIRIDIS: ColorMapValues = ColorMapValues {
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
    pub const INFERNO: ColorMapValues = ColorMapValues {
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

    pub const MAGMA: ColorMapValues = ColorMapValues {
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

    pub const PLASMA: ColorMapValues = ColorMapValues {
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

    pub const COOLWARM: ColorMapValues = ColorMapValues {
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

    pub fn get_value(&self) -> ColorMapValues {
        match self {
            ColorMap::Turbo => ColorMap::TURBO,
            ColorMap::Viridis => ColorMap::VIRIDIS,
            ColorMap::Inferno => ColorMap::INFERNO,
            ColorMap::Magma => ColorMap::MAGMA,
            ColorMap::Plasma => ColorMap::PLASMA,
            ColorMap::Coolwarm => ColorMap::COOLWARM,
        }
    }

    pub fn get_name(&self) -> &str {
        match self {
            ColorMap::Turbo => "Turbo",
            ColorMap::Viridis => "Viridis",
            ColorMap::Inferno => "Inferno",
            ColorMap::Magma => "Magma",
            ColorMap::Plasma => "Plasma",
            ColorMap::Coolwarm => "Coolwarm",
        }
    }
}

impl UiDataElement for ColorMap {
    fn draw(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_label("ColorMap")
            .selected_text(self.get_name())
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(self, ColorMap::Turbo, "Turbo")
                    .changed();
                changed |= ui
                    .selectable_value(self, ColorMap::Viridis, "Viridis")
                    .changed();
                changed |= ui
                    .selectable_value(self, ColorMap::Inferno, "Inferno")
                    .changed();
                changed |= ui
                    .selectable_value(self, ColorMap::Magma, "Magma")
                    .changed();
                changed |= ui
                    .selectable_value(self, ColorMap::Plasma, "Plasma")
                    .changed();
                changed |= ui
                    .selectable_value(self, ColorMap::Coolwarm, "Coolwarm")
                    .changed();
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
