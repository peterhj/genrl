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
  for i in (0 .. 6).rev() {
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
pub struct BinaryHeap<T> {
  data:         Vec<T>,
  leaf_idx:     usize,
  leaf_len:     usize,
  leaf_cap:     usize,
  depth_limit:  usize,
  level_limits: Vec<usize>,
}

// XXX(20151112): Using 1-based binary heap array indexing convention:
// <http://www.cse.hut.fi/en/research/SVG/TRAKLA2/tutorials/heap_tutorial/taulukkona.html>
// <https://www.cs.cmu.edu/~adamchik/15-121/lectures/Binary%20Heaps/heaps.html>

impl<T> BinaryHeap<T> {
  pub fn new(leaf_len: usize, init: T) -> BinaryHeap<T> where T: Copy {
    let leaf_cap = ceil_power2(leaf_len as u64) as usize;
    let depth_limit = 1 + log2_slow(leaf_cap as u64) as usize;
    let mut level_limits = Vec::with_capacity(depth_limit);
    for _ in 0 .. depth_limit {
      level_limits.push(0);
    }
    let mut level_offset = leaf_cap;
    level_limits[depth_limit-1] = level_offset + leaf_len;
    for d in (0 .. depth_limit-1).rev() {
      level_offset /= 2;
      level_limits[d] = level_offset + (level_limits[d+1] + 1) / 2;
    }
    let heap_sz = 2 * leaf_cap;
    let data: Vec<_> = repeat(init).take(heap_sz).collect();
    BinaryHeap{
      data:         data,
      leaf_idx:     leaf_cap,
      leaf_len:     leaf_len,
      leaf_cap:     leaf_cap,
      depth_limit:  depth_limit,
      level_limits: level_limits,
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
  pub fn level_offset(&self, depth: usize) -> usize {
    1 << depth
  }
}

#[derive(Clone)]
pub struct SlowDiscreteDist32 {
  weights:  Vec<f32>,
  cweights: Vec<f32>,
  sum:      f32,
}

impl SlowDiscreteDist32 {
  pub fn new(len: usize) -> SlowDiscreteDist32 {
    let mut weights = Vec::with_capacity(len);
    weights.resize(len, 0.0);
    let mut cweights = Vec::with_capacity(len);
    cweights.resize(len, 0.0);
    SlowDiscreteDist32{
      weights:  weights,
      cweights: cweights,
      sum:      0.0,
    }
  }

  pub fn len(&self) -> usize {
    self.weights.len()
  }

  pub fn reset(&mut self, new_weights: &[f32]) -> Result<(), ()> {
    self.weights.copy_from_slice(new_weights);
    let mut s = 0.0;
    for j in 0 .. self.weights.len() {
      s += self.weights[j];
      self.cweights[j] = s;
    }
    self.sum = s;
    Ok(())
  }

  pub fn try_sample<R>(&mut self, rng: &mut R) -> Option<usize> where R: Rng {
    if self.sum <= 0.0 {
      return None;
    }
    let u = rng.gen_range(0.0, self.sum);
    for j in 0 .. self.weights.len() {
      if u < self.cweights[j] {
        return Some(j);
      }
    }
    None
  }
}

#[derive(Clone)]
pub struct DiscreteDist32 {
  len:      usize,
  zeros:    BitVec64,
  heap:     BinaryHeap<f32>,
}

impl DiscreteDist32 {
  pub fn new(len: usize) -> DiscreteDist32 {
    let heap_sz = 2 * ceil_power2(len as u64) as usize;
    let zeros = BitVec64::with_capacity(heap_sz);
    let heap = BinaryHeap::new(len, 0.0);
    DiscreteDist32{
      len:      len,
      zeros:    zeros,
      heap:     heap,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn reset(&mut self, weights: &[f32]) -> Result<(), ()> {
    assert!(self.len >= weights.len());
    self.zeros.clear();
    /*self.heap.data[self.heap.leaf_idx .. self.heap.leaf_idx + self.len]
      .copy_from_slice(&weights);*/
    for j in 0 .. weights.len() {
      let idx = self.heap.leaf_idx + j;
      let w = weights[j];
      /*if !(w >= 0.0) {
        panic!("WARNING: bad discrete dist weight: {:e}", w);
      }*/
      if w.is_nan() || w < 0.0 {
        return Err(());
      }
      if w == 0.0 {
        self.zeros.set(idx, true);
      }
      self.heap.data[idx] = w;
    }
    for j in weights.len() .. self.heap.leaf_cap {
      let idx = self.heap.leaf_idx + j;
      self.zeros.set(idx, true);
      self.heap.data[idx] = 0.0;
    }
    for idx in (self.heap.root() .. self.heap.leaf_idx).rev() {
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      self.heap.data[idx] = self.heap.data[left_idx] + self.heap.data[right_idx];
    }
    Ok(())
  }

  pub fn zero(&mut self, j: usize) {
    assert!(j < self.len);
    let mut idx = self.heap.leaf_idx + j;
    self.zeros.set(idx, true);
    self.heap.data[idx] = 0.0;
    while idx > self.heap.root() {
      let prev_idx = idx;
      let sib_idx = self.heap.sibling(prev_idx);
      idx = self.heap.parent(prev_idx);
      if self.zeros.get(prev_idx) && self.zeros.get(sib_idx) {
        self.zeros.set(idx, true);
      }
      self.heap.data[idx] = self.heap.data[prev_idx] + self.heap.data[sib_idx];
    }
  }

  pub fn set(&mut self, j: usize, w: f32) {
    assert!(j < self.len);
    assert!(w >= 0.0);
    let mut idx = self.heap.leaf_idx + j;
    self.zeros.set(idx, w == 0.0);
    self.heap.data[idx] = w;
    while idx > self.heap.root() {
      let prev_idx = idx;
      let sib_idx = self.heap.sibling(prev_idx);
      idx = self.heap.parent(prev_idx);
      if self.zeros.get(prev_idx) && self.zeros.get(sib_idx) {
        self.zeros.set(idx, true);
      }
      self.heap.data[idx] = self.heap.data[prev_idx] + self.heap.data[sib_idx];
    }
  }

  pub fn sample<R>(&mut self, rng: &mut R) -> Option<usize> where R: Rng {
    self.try_sample(rng)
  }

  pub fn try_sample<R>(&mut self, rng: &mut R) -> Option<usize> where R: Rng {
    let mut idx = self.heap.root();
    let mut depth = 0;
    while idx < self.heap.leaf_idx {
      if self.zeros.get(idx) {
        if idx == self.heap.root() {
          return None;
        }
        while idx > self.heap.root() {
          let prev_idx = idx;
          let sib_idx = self.heap.sibling(prev_idx);
          idx = self.heap.parent(prev_idx);
          if self.zeros.get(prev_idx) && self.zeros.get(sib_idx) {
            self.zeros.set(idx, true);
          }
        }
        idx = self.heap.root();
        depth = 0;
      }
      depth += 1;
      let left_idx = self.heap.left(idx);
      let right_idx = self.heap.right(idx);
      if right_idx >= self.heap.level_limits[depth] || self.zeros.get(right_idx) {
        idx = left_idx;
        continue;
      }
      let value = self.heap.data[idx];
      let left_value = self.heap.data[left_idx];
      let u = rng.gen_range(0.0, value);
      if u < left_value {
        idx = left_idx;
      } else {
        idx = right_idx;
      }
    }
    let j = idx - self.heap.leaf_idx;
    assert!(j < self.len);
    Some(j)
  }
}
