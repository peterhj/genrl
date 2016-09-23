//use discrete::{DiscreteFilter};
//use util::{ceil_power2};

//use bit_vec::{BitVec};
use rand::{Rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::iter::{repeat};

pub fn ceil_power2(x: u64) -> u64 {
  let mut v = x;
  v -= 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  v += 1;
  v
}

pub fn log2_slow(x: u64) -> u64 {
  /*unsigned int v;  // 32-bit value to find the log2 of 
  const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
  const unsigned int S[] = {1, 2, 4, 8, 16};
  int i;

  register unsigned int r = 0; // result of log2(v) will go here
  for (i = 4; i >= 0; i--) // unroll for speed...
  {
    if (v & b[i])
    {
      v >>= S[i];
      r |= S[i];
    } 
  }*/

  const b: [u64; 6] = [0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000, 0xFFFFFFFF00000000];
  const s: [u64; 6] = [1, 2, 4, 8, 16, 32];
  let mut v = x;
  let mut r = 0;
  for i in (0 .. 5).rev() {
    if v & b[i] != 0 {
      v >>= s[i];
      r |= s[i];
    }
  }
  r
}

#[derive(Clone)]
pub struct BitVec32 {
  data:   Vec<u32>,
  length: usize,
}

impl BitVec32 {
  pub fn with_capacity(n: usize) -> BitVec32 {
    let cap = (n + 32 - 1)/32;
    let mut data = Vec::with_capacity(cap);
    for _ in 0 .. cap {
      data.push(0);
    }
    BitVec32{
      data:   data,
      length: n,
    }
  }

  pub fn as_slice(&self) -> &[u32] {
    &self.data
  }

  pub fn as_mut_slice(&mut self) -> &mut [u32] {
    &mut self.data
  }

  pub fn clear(&mut self) {
    for x in self.data.iter_mut() {
      *x = 0;
    }
  }

  pub fn get(&self, idx: usize) -> Option<bool> {
    //assert!(idx < self.length);
    if idx < self.length {
      let (word_idx, bit_idx) = (idx / 32, idx % 32);
      Some(0x0 != (0x1 & (self.data[word_idx] >> bit_idx)))
    } else {
      None
    }
  }

  pub fn set(&mut self, idx: usize, value: bool) {
    assert!(idx < self.length);
    let (word_idx, bit_idx) = (idx / 32, idx % 32);
    self.data[word_idx] &= !(0x1 << bit_idx);
    if value {
      self.data[word_idx] |= 0x1 << bit_idx;
    }
  }

  pub fn insert(&mut self, idx: usize) {
    self.set(idx, true);
  }

  pub fn remove(&mut self, idx: &usize) {
    self.set(*idx, false);
  }

  pub fn contains(&self, idx: &usize) -> bool {
    self.get(*idx).unwrap()
  }
}

#[derive(Clone)]
pub struct BitVec64 {
  data:   Vec<u64>,
  length: usize,
}

impl BitVec64 {
  pub fn with_capacity(n: usize) -> BitVec64 {
    let cap = (n + 64 - 1)/64;
    let mut data = Vec::with_capacity(cap);
    for _ in 0 .. cap {
      data.push(0);
    }
    BitVec64{
      data:   data,
      length: n,
    }
  }

  pub fn as_slice(&self) -> &[u64] {
    &self.data
  }

  pub fn as_mut_slice(&mut self) -> &mut [u64] {
    &mut self.data
  }

  pub fn clear(&mut self) {
    for x in self.data.iter_mut() {
      *x = 0;
    }
  }

  pub fn get(&self, idx: usize) -> bool {
    /*//assert!(idx < self.length);
    if idx < self.length {
      let (word_idx, bit_idx) = (idx / 64, idx % 64);
      Some(0x0 != (0x1 & (self.data[word_idx] >> bit_idx)))
    } else {
      None
    }*/
    let (word_idx, bit_idx) = (idx / 64, idx % 64);
    0x0 != (0x1 & (self.data[word_idx] >> bit_idx))
  }

  pub fn set(&mut self, idx: usize, value: bool) {
    assert!(idx < self.length);
    let (word_idx, bit_idx) = (idx / 64, idx % 64);
    self.data[word_idx] &= !(0x1 << bit_idx);
    if value {
      self.data[word_idx] |= 0x1 << bit_idx;
    }
  }

  pub fn insert(&mut self, idx: usize) {
    self.set(idx, true);
  }

  pub fn remove(&mut self, idx: &usize) {
    self.set(*idx, false);
  }

  pub fn contains(&self, idx: &usize) -> bool {
    self.get(*idx)
  }
}

#[derive(Clone)]
pub struct BHeap<T> {
  data:         Vec<T>,
  leaf_cap:     usize,
  depth_lim:    usize,
  level_lims:   Vec<usize>,
  leaf_idx:     usize,
}

// XXX(20151112): Using 1-based binary heap array indexing convention:
// <http://www.cse.hut.fi/en/research/SVG/TRAKLA2/tutorials/heap_tutorial/taulukkona.html>
// <https://www.cs.cmu.edu/~adamchik/15-121/lectures/Binary%20Heaps/heaps.html>

impl<T> BHeap<T> {
  pub fn with_capacity(leaf_len: usize, init: T) -> BHeap<T> where T: Copy {
    let leaf_cap = ceil_power2(leaf_len as u64) as usize;
    let depth_lim = 1 + log2_slow(leaf_cap as u64) as usize;
    //println!("{} {} {}", leaf_len, leaf_cap, depth_lim);
    let total_sz = 2 * leaf_cap;
    let data: Vec<_> = repeat(init).take(total_sz).collect();
    let mut level_lims = Vec::with_capacity(depth_lim);
    for _ in 0 .. depth_lim {
      level_lims.push(0);
    }
    let mut level_offset = leaf_cap;
    level_lims[depth_lim-1] = level_offset + leaf_len;
    for d in (0 .. depth_lim-1).rev() {
      //println!("DEBUG: {}/{}", d, depth_lim);
      level_offset /= 2;
      level_lims[d] = level_offset + (level_lims[d+1] + 1) / 2;
    }
    BHeap{
      data:         data,
      leaf_cap:     leaf_cap,
      depth_lim:    depth_lim,
      level_lims:   level_lims,
      leaf_idx:     leaf_cap,
    }
  }

  #[inline]
  pub fn root(&self) -> usize {
    1
  }

  #[inline]
  pub fn parent(&self, idx: usize) -> usize {
    idx / 2
  }

  #[inline]
  pub fn sibling(&self, idx: usize) -> usize {
    //2 * (idx / 2) + 1 - (idx % 2)
    idx + 1 - 2 * (idx % 2)
  }

  #[inline]
  pub fn left(&self, idx: usize) -> usize {
    2 * idx
  }

  #[inline]
  pub fn right(&self, idx: usize) -> usize {
    2 * idx + 1
  }

  #[inline]
  pub fn level(&self, depth: usize) -> usize {
    1 << depth
  }
}

#[derive(Clone)]
pub struct DiscreteDist32 {
  len:      usize,
  heap:     BHeap<f32>,
  zeros:    BitVec64,
  //range:    Range<f32>,
}

impl DiscreteDist32 {
  pub fn new(len: usize) -> DiscreteDist32 {
    let heap = BHeap::with_capacity(len, 0.0);
    let heap_len = heap.data.len();
    DiscreteDist32{
      len:      len,
      heap:     heap,
      zeros:    BitVec64::with_capacity(heap_len),
      //range:    Range::new(0.0, 1.0),
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn reset(&mut self, weights: &[f32]) {
    assert_eq!(self.len, weights.len());
    self.zeros.clear();
    self.heap.data[self.heap.leaf_idx .. self.heap.leaf_idx + self.len]
      .copy_from_slice(&weights);
    for j in self.len .. self.heap.leaf_cap {
      let idx = self.heap.leaf_idx + j;
      self.heap.data[idx] = 0.0;
    }
    for idx in (self.heap.root() .. self.heap.leaf_idx).rev() {
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      self.heap.data[idx] = self.heap.data[left_idx] + self.heap.data[right_idx];
      self.zeros.set(idx, true);
    }
  }

  pub fn zero(&mut self, j: usize) {
    assert!(j < self.len);
    let idx = self.heap.leaf_idx + j;
    self.heap.data[idx] = 0.0;
    self.zeros.set(idx, true);
  }

  pub fn sample<R>(&mut self, rng: &mut R) -> Option<usize> where R: Rng {
    let mut idx = self.heap.root();
    let mut depth = 0;
    while idx < self.heap.leaf_idx {
      if self.zeros.get(idx) {
        if idx == self.heap.root() {
          return None;
        }
        while idx > self.heap.root() {
          let prev_idx = idx;
          idx = self.heap.parent(idx);
          if self.zeros.get(prev_idx) && self.zeros.get(self.heap.sibling(prev_idx)) {
            self.zeros.set(idx, true);
          }
        }
        depth = 0;
      }
      depth += 1;
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      if right_idx >= self.heap.level_lims[depth] {
        idx = left_idx;
        continue;
      }
      let value = self.heap.data[idx];
      let left_value = self.heap.data[left_idx];
      //let u = value * self.range.ind_sample(rng);
      let u = rng.gen_range(0.0, value);
      if u >= left_value && !self.zeros.get(right_idx) {
        idx = right_idx;
      } else {
        idx = left_idx;
      }
    }
    Some(idx - self.heap.leaf_idx)
  }
}

#[derive(Clone)]
pub struct DiscreteSampler {
  max_n:    usize,
  leaf_idx: usize,
  n:        usize,
  heap:     BHeap<f32>,
  marks:    BitVec32,
  zeros:    BitVec32,
}

impl DiscreteSampler {
  pub fn with_capacity(max_n: usize) -> DiscreteSampler {
    let heap = BHeap::with_capacity(max_n, 0.0);
    //let half_cap = ceil_power2(n as u64) as usize - 1;
    //let cap = 2 * ceil_power2(n as u64) as usize - 1;
    let half_cap = ceil_power2(max_n as u64) as usize;
    let cap = 2 * ceil_power2(max_n as u64) as usize;
    let marks = BitVec32::with_capacity(2 * half_cap);
    let zeros = BitVec32::with_capacity(cap - half_cap);
    DiscreteSampler{
      max_n:    max_n,
      leaf_idx: half_cap,
      n:        0,
      heap:     heap,
      marks:    marks,
      zeros:    zeros,
    }
  }

  #[inline]
  fn mark_idx(&self, idx: usize) -> usize {
    //2 * self.heap.parent(idx) + 1 - (idx % 2)
    2 * self.heap.parent(idx) + (idx % 2)
  }

  #[inline]
  fn mark_sibling_idx(&self, idx: usize) -> usize {
    //2 * self.heap.parent(idx) + (idx % 2)
    2 * self.heap.parent(idx) + 1 - (idx % 2)
  }

  #[inline]
  fn mark_left_idx(&self, parent_idx: usize) -> usize {
    2 * parent_idx
  }

  #[inline]
  fn mark_right_idx(&self, parent_idx: usize) -> usize {
    2 * parent_idx + 1
  }

  pub fn capacity(&self) -> usize {
    self.max_n
  }

  pub fn unsafe_reset(&mut self) {
    self.n = self.max_n;
  }

  pub fn get_mut_parts(&mut self) -> (&mut [f32], &mut [u32], &mut [u32]) {
    (&mut self.heap.data, &mut self.marks.data, &mut self.zeros.data)
  }

  pub fn reset(&mut self, xs: &[f32]) {
    assert_eq!(self.max_n, xs.len());
    self.n = self.max_n;
    self.marks.clear();
    self.zeros.clear();

    (&mut self.heap.data[self.leaf_idx .. self.leaf_idx + self.n]).clone_from_slice(xs);

    // XXX: Remaining part of heap data should always be zeroed.
    for idx in self.leaf_idx + self.n .. self.heap.data.len() {
      let mark_idx = self.mark_idx(idx);
      self.marks.set(mark_idx, true);
      self.zeros.set(idx - self.leaf_idx, true);
    }
    for idx in (self.heap.root() .. self.leaf_idx).rev() {
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      self.heap.data[idx] = self.heap.data[left_idx] + self.heap.data[right_idx];
      if self.marks.get(self.mark_left_idx(idx)).unwrap() && self.marks.get(self.mark_right_idx(idx)).unwrap() {
        if idx > 0 {
          let mark_idx = self.mark_idx(idx);
          self.marks.set(mark_idx, true);
        }
      }
    }
  }

  pub fn zero(&mut self, j: usize) {
    if self.zeros.get(j).unwrap() {
      return;
    }
    self.n -= 1;
    self.zeros.set(j, true);
    if self.n == 0 {
      return;
    }

    let root = self.heap.root();
    let mut idx = self.leaf_idx + j;
    let mut do_mark = true;
    self.heap.data[idx] = 0.0;
    loop {
      if idx == root {
        break;
      }
      let parent_idx = self.heap.parent(idx);
      // XXX: Note: subtracting the zeroed out value, instead of recomputing the
      // sum, leads to roundoff errors which seriously mess up the sampling part
      // of the data structure.
      self.heap.data[parent_idx] = self.heap.data[idx] + self.heap.data[self.heap.sibling(idx)];
      if do_mark {
        let mark_idx = self.mark_idx(idx);
        let mark_sib_idx = self.mark_sibling_idx(idx);
        self.marks.set(mark_idx, true);
        if !self.marks.get(mark_sib_idx).unwrap() {
          do_mark = false;
        }
      }
      idx = parent_idx;
    }
  }

  pub fn sample<R: Rng>(&mut self, rng: &mut R) -> Option<usize> {
    if self.n == 0 {
      return None;
    }
    let mut idx = self.heap.root();
    loop {
      if idx >= self.leaf_idx {
        break;
      }
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      match (self.marks.get(self.mark_left_idx(idx)).unwrap(), self.marks.get(self.mark_right_idx(idx)).unwrap()) {
        (false, false) => {
          let value = self.heap.data[idx];
          //assert!(value > 0.0); // XXX: Range already checks this.
          //println!("DEBUG: gen_range value: {}", value);
          if !(value > 0.0) {
            fn bfs_zero(filter: &mut DiscreteSampler, idx: usize) {
              let leaf_idx = filter.leaf_idx;
              if idx >= leaf_idx {
                filter.zero(idx - leaf_idx);
              } else {
                let left_idx = filter.heap.left(idx);
                let right_idx = filter.heap.right(idx);
                bfs_zero(filter, left_idx);
                bfs_zero(filter, right_idx);
              }
            };
            bfs_zero(self, idx);
            return self.sample(rng);
          }
          let left_value = self.heap.data[left_idx];
          let u = rng.gen_range(0.0, value);
          if u < left_value {
            idx = left_idx;
          } else {
            idx = right_idx;
          }
        }
        (false, true) => {
          idx = left_idx;
        }
        (true, false) => {
          idx = right_idx;
        }
        _ => {
          unreachable!();
        }
      }
    }
    Some(idx - self.leaf_idx)
  }
}
