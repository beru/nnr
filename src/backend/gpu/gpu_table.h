#pragma once
// gpu_table — SoA (Structure of Arrays) container on device.
//
// Manages multiple typed arrays that share the same row count.
// Each column is a separate contiguous device allocation — optimal
// for GPU coalesced access and SIMD-style processing.
//
// Usage:
//   // Define columns
//   gpu_table table(device, 1000000);       // 1M rows
//   auto& x = table.add_column<float>("x");
//   auto& y = table.add_column<float>("y");
//   auto& z = table.add_column<float>("z");
//   auto& id = table.add_column<int>("id");
//
//   // Access
//   float* px = table.column<float>("x").data();
//   size_t rows = table.rows();
//
//   // Resize all columns together
//   table.resize(2000000);

#include "gpu_array.h"
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <stdexcept>

namespace nnr::gpu {

// Type-erased column base
struct column_base {
    virtual ~column_base() = default;
    virtual void resize(size_t new_rows) = 0;
    virtual size_t elem_size() const = 0;
    virtual void* raw_data() = 0;
};

// Typed column
template <typename T>
struct column_t : column_base {
    gpu_array<T> array;

    column_t(gpu_device_t* dev, size_t rows) : array(dev, rows) {}

    void resize(size_t new_rows) override { array.resize(new_rows); }
    size_t elem_size() const override { return sizeof(T); }
    void* raw_data() override { return array.data(); }
};

struct gpu_table {
    gpu_table() = default;
    gpu_table(gpu_device_t* dev, size_t rows) : device_(dev), rows_(rows) {}

    // --- Add column ---

    template <typename T>
    gpu_array<T>& add_column(std::string_view name) {
        auto col = std::make_unique<column_t<T>>(device_, rows_);
        auto& ref = col->array;
        names_.push_back(std::string(name));
        columns_.push_back(std::move(col));
        return ref;
    }

    // --- Access column by name ---

    template <typename T>
    gpu_array<T>& column(std::string_view name) {
        for (size_t i = 0; i < names_.size(); i++) {
            if (names_[i] == name)
                return static_cast<column_t<T>*>(columns_[i].get())->array;
        }
        throw std::runtime_error("column not found");
    }

    template <typename T>
    const gpu_array<T>& column(std::string_view name) const {
        for (size_t i = 0; i < names_.size(); i++) {
            if (names_[i] == name)
                return static_cast<const column_t<T>*>(columns_[i].get())->array;
        }
        throw std::runtime_error("column not found");
    }

    // --- Access column by index ---

    template <typename T>
    gpu_array<T>& column(size_t index) {
        return static_cast<column_t<T>*>(columns_[index].get())->array;
    }

    // --- Properties ---

    size_t rows() const { return rows_; }
    size_t num_columns() const { return columns_.size(); }
    std::string_view column_name(size_t i) const { return names_[i]; }

    // --- Resize all columns ---

    void resize(size_t new_rows) {
        for (auto& col : columns_)
            col->resize(new_rows);
        rows_ = new_rows;
    }

    // --- Device ---

    gpu_device_t* device() const { return device_; }

private:
    gpu_device_t* device_ = nullptr;
    size_t rows_ = 0;
    std::vector<std::string> names_;
    std::vector<std::unique_ptr<column_base>> columns_;
};

} // namespace nnr::gpu
