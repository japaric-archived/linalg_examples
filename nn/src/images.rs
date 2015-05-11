use std::fs::File;
use std::io::{Read, Seek, SeekFrom, self};
use std::iter;
use std::ops::Range;
use std::path::Path;

use byteorder::{BigEndian, ReadBytesExt};
use cast::From as _0;
use image::{ImageBuffer, Luma};
use linalg::prelude::*;

/// A set of images in compressed format
pub struct Images {
    data: Vec<u8>,
    height: u32,
    img_size: usize,
    size: u32,
    width: u32,
}

impl Images {
    /// Loads a subset of the images stored in `path`
    pub fn load<P>(path: P, subset: Range<u32>) -> io::Result<Images> where P: AsRef<Path> {
        Images::load_(path.as_ref(), subset)
    }

    fn load_(path: &Path, Range { start, end }: Range<u32>) -> io::Result<Images> {
        /// Magic number expected in the header
        const MAGIC: u32 = 2051;

        assert!(start < end);

        let mut file = try!(File::open(path));

        // Parse the header: MAGIC NIMAGES NROWS NCOLS
        assert_eq!(try!(file.read_u32::<BigEndian>()), MAGIC);
        let nimages = try!(file.read_u32::<BigEndian>());
        let nrows = try!(file.read_u32::<BigEndian>());
        let ncols = try!(file.read_u32::<BigEndian>());

        assert!(end <= nimages);

        let img_size = usize::from_(nrows).checked_mul(usize::from_(ncols)).unwrap();
        let buf_size = img_size * usize::from_(end - start);
        let mut buf: Vec<_> = iter::repeat(0).take(buf_size).collect();

        try!(file.seek(SeekFrom::Current(i64::from_(img_size * usize::from_(start)).unwrap())));

        assert_eq!(try!(file.read(&mut buf)), buf_size);

        Ok(Images {
            data: buf,
            height: nrows,
            img_size: img_size,
            size: end - start,
            width: ncols,
        })
    }

    /// Returns the number of pixes per image
    pub fn num_pixels(&self) -> u32 {
        self.width * self.height
    }

    /// Returns the size of this set
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Return the images as a data set that can be fed to a neural network
    ///
    /// The returned matrix has dimensions m-by-(n+1), where
    ///
    /// m: Number of images
    /// n: Number of pixels per image
    ///
    /// Each row of the matrix is an "unrolled" image where each element of the row represents the
    /// brightness (in the 0.0 - 1.0 range) of a single pixel
    pub fn to_dataset(&self) -> Mat<f64> {
        let mut m = Mat::ones((self.size, self.width * self.height + 1));

        for (mut row, img) in m.rows_mut().zip(self.data.chunks(self.img_size)) {
            for (e, &brightness) in row.iter_mut().skip(1).zip(img) {
                *e = f64::from_(brightness) / f64::from_(u8::max_value())
            }
        }

        m
    }

    /// Saves `these` images (up to 100) in `path`
    pub fn save<I, P>(&self, these: I, path: P) -> io::Result<()> where
        I: IntoIterator<Item=u32>,
        P: AsRef<Path>,
    {
        self.save_(these.into_iter(), path.as_ref())
    }

    fn save_<I>(&self, mut these: I, path: &Path) -> io::Result<()> where
        I: Iterator<Item=u32>
    {
        const NROWS: u32 = 10;
        const NCOLS: u32 = 10;

        let mut buf: Vec<_> =
            iter::repeat(0).take(usize::from_(NCOLS * NROWS) * self.img_size).collect();

        // Too bad that linalg's block copying doesn't work with the u8 type...
        'out: for r in 0..NROWS {
            for c in 0..NCOLS {
                let i = match these.next() {
                    None => break 'out,
                    Some(i) => i,
                };

                assert!(i < self.size);

                let start = usize::from_(i) * self.img_size;
                let end = start + self.img_size;

                let img = &self.data[start..end];

                for i in 0..self.height {
                    for j in 0..self.width {
                        let k = usize::from_(i * self.width + j);

                        let ii = r * self.height + i;
                        let jj = c * self.width + j;
                        let kk = usize::from_(ii * NCOLS * self.width + jj);

                        buf[kk] = img[k];
                    }
                }
            }
        }

        let width = NCOLS * self.width;
        let height = NROWS * self.height;

        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, buf).unwrap();

        img.save(path)
    }
}
