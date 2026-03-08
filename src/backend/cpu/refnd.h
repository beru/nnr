#pragma once

template <typename T>
class ref1d
{
public:
  ref1d(T* data = nullptr)
    :
    data(data)
  {}
  
  T*& operator() () { return data; }
  T& operator [] (size_t idx) { return data[idx]; }
    
private:
  T* data;
};

template <typename T>
class ref2d
{
public:
  ref2d(size_t dim1, T* data = nullptr)
    :
    dim1(dim1),
    data(data)
  {}

  T*& operator() () { return data; }
  ref1d<T> operator [] (size_t idx) { return ref1d(&data[idx * dim1]); }
    
private:
  T* data;
  size_t dim1;
};

template <typename T>
class ref3d
{
public:
  ref3d(size_t dim1, size_t dim2, T* data = nullptr)
    :
    dim1(dim1),
    dim2(dim2),
    data(data)
  {}
  
  T*& operator() () { return data; }
  ref2d<T> operator [] (size_t idx) { return ref2d(dim1, &data[idx * dim1 * dim2]); }
    
private:
  T* data;
  size_t dim1;
  size_t dim2;
};

template <typename T>
class ref4d
{
public:
  ref4d(size_t dim1, size_t dim2, size_t dim3, T* data = nullptr)
    :
    dim1(dim1),
    dim2(dim2),
    dim3(dim3),
    data(data)
  {}
  
  T*& operator() () { return data; }
  ref3d<T> operator [] (size_t idx) { return ref3d(dim1, dim2, &data[idx * dim1 * dim2 * dim3]); }
    
private:
  T* data;
  size_t dim1;
  size_t dim2;
  size_t dim3;
};

