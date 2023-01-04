# deuxfleurs

Viewer for geometry processing / meshes / 3d related stuff heavily inspired by [polyscope](https://polyscope.run) (which is very nice, check it out!).
Can be used in webpages thanks to wasm (such as reveal.js slides). Current repo can be used with [trunk](https://github.com/thedodd/trunk), just run `trunk serve`.

Currently hardly usable, notably because of the lack of genericity on the types used for the meshes : only a couple `(Vec<[f32; 3]>, Vec<[usize; 3]>)` can be used as a mesh descriptor, no compatibility yet with types from `nalgbra` or `ndarray`.
Also only supports triangular surface mesh for the moment.

A few things that should be added :
* genericity over input types
* other data structures such as polygonal meshes, vector fields, point clouds, volumes
* state saving/loading
* documentation
