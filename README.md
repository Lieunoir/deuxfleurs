# Deuxfleurs
[![Documentation][doc-img]][doc-url]

[doc-img]: https://img.shields.io/badge/doc-deuxfleurs-green
[doc-url]: https://lieunoir.github.io/deuxfleurs/deuxfleurs/

![screenshot_000](https://github.com/user-attachments/assets/450e9d86-f8de-419d-9de4-f329545a97e2)

Viewer for geometry processing / meshes / 3d related stuff heavily inspired by [polyscope](https://polyscope.run) (which is very nice, check it out!).

Can be used in webpages thanks to wasm, so web-based slides can include demos, or simply for making code demo easily accessible on the web.

Current repo can be used with [trunk](https://github.com/thedodd/trunk), just run `trunk serve`.

# How to use

Here's a quick example that loads a mesh and uses a button to show/hide it.
```rust
use deuxfleurs::{load_mesh, State, StateBuilder, Ui};
// Load the mesh and register it in state:
let (v, f) = load_mesh("bunny.obj").await.unwrap();
let init = move |state: &mut State| {
    state.register_surface("bunny".into(), v, f);
};

// Toggle between shown or not on button pressed
let callback = |state: &mut State, ui: &mut Ui| {
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
```
