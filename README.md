# Deuxfleurs
[![Documentation][doc-img]][doc-url]

[doc-img]: https://img.shields.io/badge/doc-deuxfleurs-green
[doc-url]: https://lieunoir.github.io/deuxfleurs/deuxfleurs/

![screenshot_000](https://github.com/user-attachments/assets/450e9d86-f8de-419d-9de4-f329545a97e2)

Viewer for geometry processing / meshes / 3d related stuff heavily inspired by [polyscope](https://polyscope.run) (which is very nice, check it out!).

Can be used in webpages thanks to wasm, so web-based slides can include demos, or simply for making code demo easily accessible on the web. An example can be found [here](https://github.com/Lieunoir/uvat-wasm-demo).

# How to use

Here's a quick example that loads a mesh and uses a button to show/hide it:
```rust
use deuxfleurs::egui;
use deuxfleurs::{load_mesh, RunningState, StateHandle, Settings};
// Init the app
let mut handle = deuxfleurs::init();
// Load the mesh and register it:
let (v, f) = load_mesh("bunny.obj").await.unwrap();
handle.register_surface("bunny".into(), v, f);

// Toggle between shown or not on button pressed
let callback = |ui: &mut egui::Ui, state: &mut RunningState| {
    if ui
        .add(egui::Button::new("Toggle shown"))
        .clicked()
    {
        let mut surface = state.get_surface_mut("bunny").unwrap();
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
```

More examples are available in the corresponding folder.
