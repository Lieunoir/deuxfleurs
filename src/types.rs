#[derive(Clone, PartialEq)]
pub enum SurfaceIndices {
    Triangles(Vec<[u32; 3]>),
    Quads(Vec<[u32; 4]>),
    Polygons(Vec<u32>, Vec<u32>),
}

impl SurfaceIndices {
    /// Number of faces
    pub fn size(&self) -> usize {
        match self {
            SurfaceIndices::Triangles(t) => t.len(),
            SurfaceIndices::Quads(q) => q.len(),
            SurfaceIndices::Polygons(_i, s) => s.len() - 1,
        }
    }

    /// Numer of triangles once each face is split in triangles
    pub fn tot_triangles(&self) -> usize {
        match self {
            SurfaceIndices::Triangles(t) => t.len(),
            SurfaceIndices::Quads(q) => 2 * q.len(),
            SurfaceIndices::Polygons(_i, s) => s[s.len() - 1] as usize - (2 * (s.len() - 1)),
        }
    }
}

impl std::ops::Index<usize> for SurfaceIndices {
    type Output = [u32];

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            SurfaceIndices::Triangles(t) => &t[index],
            SurfaceIndices::Quads(q) => &q[index],
            SurfaceIndices::Polygons(i, s) => &i[s[index] as usize..s[index + 1] as usize],
        }
    }
}

impl Into<SurfaceIndices> for Vec<[u32; 3]> {
    fn into(self) -> SurfaceIndices {
        SurfaceIndices::Triangles(self)
    }
}

impl Into<SurfaceIndices> for Vec<[u32; 4]> {
    fn into(self) -> SurfaceIndices {
        SurfaceIndices::Quads(self)
    }
}

impl Into<SurfaceIndices> for (Vec<u32>, Vec<u32>) {
    fn into(self) -> SurfaceIndices {
        let mut count = 0;
        let mut faces_indices = self
            .1
            .into_iter()
            .map(|s| {
                count += s;
                count - s
            })
            .collect::<Vec<_>>();
        faces_indices.push(count);
        SurfaceIndices::Polygons(self.0, faces_indices)
    }
}

impl Into<SurfaceIndices> for (Vec<u32>, Vec<u8>) {
    fn into(self) -> SurfaceIndices {
        let mut count = 0;
        let mut faces_indices = self
            .1
            .into_iter()
            .map(|s| {
                count += s as u32;
                count - s as u32
            })
            .collect::<Vec<_>>();
        faces_indices.push(count);
        SurfaceIndices::Polygons(self.0, faces_indices)
    }
}

/// Helper iterator struct
// Could be more efficient
pub struct SurfaceIndicesIntoIterator<'a> {
    indices: &'a SurfaceIndices,
    index: usize,
}

impl<'a> IntoIterator for &'a SurfaceIndices {
    type Item = &'a [u32];
    type IntoIter = SurfaceIndicesIntoIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SurfaceIndicesIntoIterator {
            indices: self,
            index: 0,
        }
    }
}

use std::borrow::Borrow;

impl<'a> Iterator for SurfaceIndicesIntoIterator<'a> {
    type Item = &'a [u32];

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self.indices {
            SurfaceIndices::Triangles(t) => t.len() - self.index,
            SurfaceIndices::Quads(q) => q.len() - self.index,
            SurfaceIndices::Polygons(_, s) => s.len() - self.index - 1,
        };
        (len, Some(len))
    }

    fn next(&mut self) -> Option<&'a [u32]> {
        let res = match self.indices {
            SurfaceIndices::Triangles(t) => t.get(self.index).map(|a| a.borrow()),
            SurfaceIndices::Quads(q) => q.get(self.index).map(|a| a.borrow()),
            SurfaceIndices::Polygons(i, s) => s
                .get(self.index)
                .map(|offset| {
                    s.get(self.index + 1)
                        .map(|offset2| i.get(*offset as usize..*offset2 as usize))
                })
                .flatten()
                .flatten(),
        };
        self.index += 1;
        res
    }
}

impl<'a> ExactSizeIterator for SurfaceIndicesIntoIterator<'a> {
    fn len(&self) -> usize {
        match self.indices {
            SurfaceIndices::Triangles(t) => t.len() - self.index,
            SurfaceIndices::Quads(q) => q.len() - self.index,
            SurfaceIndices::Polygons(_, s) => s.len() - self.index - 1,
        }
    }
}

pub trait Vertices {
    fn into(self) -> Vec<[f32; 3]>;
}

impl Vertices for Vec<[f32; 3]> {
    fn into(self) -> Vec<[f32; 3]> {
        self
    }
}

impl Vertices for &[[f64; 3]] {
    fn into(self) -> Vec<[f32; 3]> {
        self.iter()
            .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
            .collect()
    }
}

impl Vertices for &Vec<[f64; 3]> {
    fn into(self) -> Vec<[f32; 3]> {
        self.iter()
            .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
            .collect()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
impl Vertices for ndarray::Array2<f32> {
    fn into(self) -> Vec<[f32; 3]> {
        self.rows()
            .into_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
impl Vertices for ndarray::Array2<f64> {
    fn into(self) -> Vec<[f32; 3]> {
        self.rows()
            .into_iter()
            .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
            .collect()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
impl Vertices for nalgebra::base::MatrixXx3<f32> {
    fn into(self) -> Vec<[f32; 3]> {
        self.row_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }
}

pub trait Vertices2D {
    fn into(self) -> Vec<[f32; 2]>;
}

impl Vertices2D for Vec<[f32; 2]> {
    fn into(self) -> Vec<[f32; 2]> {
        self
    }
}

impl Vertices2D for &[[f64; 2]] {
    fn into(self) -> Vec<[f32; 2]> {
        self.iter()
            .map(|row| [row[0] as f32, row[1] as f32])
            .collect()
    }
}

impl Vertices2D for &Vec<[f64; 2]> {
    fn into(self) -> Vec<[f32; 2]> {
        self.iter()
            .map(|row| [row[0] as f32, row[1] as f32])
            .collect()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
impl Vertices2D for ndarray::Array2<f32> {
    fn into(self) -> Vec<[f32; 2]> {
        self.rows()
            .into_iter()
            .map(|row| [row[0], row[1]])
            .collect()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
impl Vertices2D for nalgebra::base::MatrixXx3<f32> {
    fn into(self) -> Vec<[f32; 2]> {
        self.row_iter().map(|row| [row[0], row[1]]).collect()
    }
}

pub trait Scalar {
    fn into(self) -> Vec<f32>;
}

impl Scalar for Vec<f32> {
    fn into(self) -> Vec<f32> {
        self
    }
}

impl Scalar for &[f64] {
    fn into(self) -> Vec<f32> {
        self.iter().map(|row| *row as f32).collect()
    }
}

impl Scalar for &Vec<f64> {
    fn into(self) -> Vec<f32> {
        self.iter().map(|row| *row as f32).collect()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
impl Scalar for ndarray::Array1<f32> {
    fn into(self) -> Vec<f32> {
        self.into_raw_vec()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
impl Scalar for ndarray::Array1<f64> {
    fn into(self) -> Vec<f32> {
        self.into_raw_vec().into_iter().map(|f| f as f32).collect()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
impl Scalar for nalgebra::base::DVector<f32> {
    fn into(self) -> Vec<f32> {
        self.row_iter().map(|row| row[0]).collect()
    }
}

pub trait Color {
    fn into(self) -> Vec<[f32; 3]>;
}

impl Color for &Vec<[f64; 3]> {
    fn into(self) -> Vec<[f32; 3]> {
        self.iter()
            .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
            .collect()
    }
}

impl Color for &[[f64; 3]] {
    fn into(self) -> Vec<[f32; 3]> {
        self.iter()
            .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
            .collect()
    }
}

impl Color for Vec<[f32; 3]> {
    fn into(self) -> Vec<[f32; 3]> {
        self
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
impl Color for ndarray::Array2<f32> {
    fn into(self) -> Vec<[f32; 3]> {
        self.rows()
            .into_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
#[cfg(feature = "ndarray")]
impl Color for ndarray::Array2<f64> {
    fn into(self) -> Vec<[f32; 3]> {
        self.rows()
            .into_iter()
            .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
            .collect()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
impl Color for nalgebra::base::MatrixXx3<f32> {
    fn into(self) -> Vec<[f32; 3]> {
        self.row_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }
}
