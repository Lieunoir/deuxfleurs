use deuxfleurs::egui;
use deuxfleurs::{load_mesh, Color, State, StateBuilder};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

fn main() {
    pollster::block_on(run());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    // Load the mesh and register it in state:
    let (v, f) = load_mesh("examples/assets/bunnyhead.obj").await.unwrap();
    let init = move |state: &mut State| {
        state.register_surface("bunny".into(), v, f);
    };

    // Toggle between shown or not on button pressed
    let callback = |ui: &mut egui::Ui, state: &mut State| {
        if ui.add(egui::Button::new("Toggle shown")).clicked() {
            let surface = state.get_surface_mut("bunny").unwrap();
            let shown = surface.shown();
            surface.show(!shown);
        }
    };
    // Run the app
    StateBuilder::run(
        1080,
        720,
        Some("deuxfleurs-demo".into()),
        deuxfleurs::Settings {
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            ..Default::default()
        },
        init,
        callback,
    );
}
