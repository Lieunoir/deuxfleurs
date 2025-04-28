use deuxfleurs::egui;
use deuxfleurs::{load_mesh, RunningState, Settings, StateHandle};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

fn main() {
    pollster::block_on(run());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    // Initialize app
    let mut handle = deuxfleurs::init();

    // Load the mesh and register it in state:
    let (v, f) = load_mesh("examples/assets/bunnyhead.obj").await.unwrap();
    handle.register_surface("bunny".into(), v, f);

    // Toggle between shown or not on button pressed
    let callback = |ui: &mut egui::Ui, state: &mut RunningState| {
        if ui.add(egui::Button::new("Toggle shown")).clicked() {
            let surface = state.get_surface_mut("bunny").unwrap();
            let shown = surface.shown();
            surface.show(!shown);
        }
    };
    // Run the app
    handle.run(
        1080,
        720,
        Some("deuxfleurs-demo".into()),
        Settings::default(),
        callback,
    );
}
